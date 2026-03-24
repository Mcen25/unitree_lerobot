"""
OpenVLA inference server — run this on the Jetson Thor (or any CUDA machine).

Loads openvla-7b + LoRA fine-tune adapter, listens for image+task requests
over ZMQ, and returns 7-DOF delta EE actions to the robot.

Usage (Jetson Thor):
    conda activate <env_with_transformers_peft>
    python openvla_server.py \\
        --checkpoint /path/to/Checkpoints/OpenVLA/openvla-7b+pick_n_place_orange_bottle+... \\
        --port 5555

    # If base model is already cached locally, pass the snapshot path:
    python openvla_server.py \\
        --checkpoint /path/to/checkpoint \\
        --base-model /home/user/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/<hash> \\
        --port 5555

Notes for Jetson Thor (aarch64):
  - flash_attention_2 is NOT used (not available on aarch64) — uses eager attention
  - pixel_values are cast to float32 (required by OpenVLA vision encoder)
  - model weights loaded in float16 by default (stable on all Jetson variants)
  - 64 GB unified memory is sufficient for openvla-7b in float16 (~14 GB)
"""

import argparse
import base64
import json
import os
import traceback
from io import BytesIO

import numpy as np
import zmq
from PIL import Image


# ---------------------------------------------------------------------------
# OpenVLA prompt template
# ---------------------------------------------------------------------------

def make_prompt(task: str) -> str:
    return f"In: What action should the robot take to {task}?\nOut:"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA inference server")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned OpenVLA checkpoint directory (contains lora_adapter/ and dataset_statistics.json). "
             "Omit to run the base model without a LoRA adapter.",
    )
    p.add_argument(
        "--base-model",
        default="openvla/openvla-7b",
        help="Base model: HuggingFace ID or local snapshot path",
    )
    p.add_argument(
        "--unnorm-key",
        default="bridge_orig",
        help="Action unnormalization key to use when running without a checkpoint (default: bridge_orig)",
    )
    p.add_argument("--port", type=int, default=5555, help="ZMQ REP port")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype. float16 is recommended for Jetson aarch64.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(checkpoint_path: str | None, base_model: str, dtype_str: str):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server] Device: {device}")

    processor_src = checkpoint_path if checkpoint_path else base_model
    print(f"[server] Loading processor from {processor_src} ...")
    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)

    print(f"[server] Loading base model from '{base_model}' ...")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # flash_attention_2 is NOT available on aarch64 — use eager
        attn_implementation="eager",
    )

    if checkpoint_path:
        from peft import PeftModel
        lora_path = os.path.join(checkpoint_path, "lora_adapter")
        print(f"[server] Merging LoRA adapter from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

        # Inject fine-tuned dataset normalization stats into the model.
        # The base model only contains its original training dataset stats;
        # the fine-tuned key (e.g. "pick_n_place_orange_bottle") must be added explicitly.
        stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
        with open(stats_path) as f:
            custom_stats = json.load(f)
        model.norm_stats.update(custom_stats)
        print(f"[server] Injected norm_stats keys: {list(custom_stats.keys())}")
    else:
        print("[server] No checkpoint provided — running base model without LoRA adapter.")

    model = model.to(device)
    model.eval()

    print(f"[server] Model ready — {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params, dtype={dtype}")
    return model, processor, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    image: Image.Image,
    task: str,
    unnorm_key: str,
    device: str,
) -> list:
    """
    One forward pass of OpenVLA.
    Returns unnormalized 7-DOF action: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]

    Important: pixel_values MUST be float32 for OpenVLA's vision encoder,
    regardless of the model's weight dtype.
    """
    import torch

    prompt = make_prompt(task)

    # processor returns dict with input_ids, attention_mask, pixel_values
    inputs = processor(prompt, image)

    # Float tensors (pixel_values) → model dtype; integer tensors (input_ids) → device only
    inputs = {
        k: v.to(device, dtype=model.dtype) if (hasattr(v, "to") and v.is_floating_point())
        else v.to(device) if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    # Drop attention_mask: the vision encoder expands sequence length beyond the
    # text-only mask size, causing a causal mask mismatch during generation.
    # With batch_size=1 and no padding, attention_mask is not needed.
    inputs_for_pred = {k: v for k, v in inputs.items() if k != "attention_mask"}

    import numpy as _np
    _pv = inputs_for_pred.get("pixel_values")
    if _pv is not None:
        _arr = _pv.float().cpu().numpy()
        print(f"[dbg] pixel_values sum={_arr.sum():.1f} mean={_arr.mean():.4f} std={_arr.std():.4f}")

    with torch.no_grad():
        # Get raw predicted token IDs to verify model is producing varied outputs
        _raw = model.generate(
            **inputs_for_pred,
            max_new_tokens=7,
            do_sample=False,
        )
        _new_tokens = _raw[0, -7:].tolist()
        print(f"[dbg] raw token IDs: {_new_tokens}")
        action = model.predict_action(**inputs_for_pred, unnorm_key=unnorm_key, do_sample=False)

    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    return [float(x) for x in action]


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model, processor, device = load_model_and_processor(
        args.checkpoint, args.base_model, args.dtype
    )

    # Determine unnorm_key: from checkpoint stats if available, else from --unnorm-key arg
    if args.checkpoint:
        stats_path = os.path.join(args.checkpoint, "dataset_statistics.json")
        with open(stats_path) as f:
            dataset_stats = json.load(f)
        unnorm_key = list(dataset_stats.keys())[0]
    else:
        unnorm_key = args.unnorm_key
    print(f"[server] unnorm_key = '{unnorm_key}'")

    # ZMQ REP socket
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[server] Listening on port {args.port} ...")

    step = 0
    while True:
        try:
            msg = socket.recv_json()
            task = msg.get("task", unnorm_key)
            image = Image.open(BytesIO(base64.b64decode(msg["image"]))).convert("RGB")

            action = run_inference(model, processor, image, task, unnorm_key, device)

            if step % 20 == 0:
                print(f"[server] step={step} | action={np.round(action, 4)}")
            step += 1

            socket.send_json({"action": action, "status": "ok"})

        except KeyboardInterrupt:
            print("[server] Shutting down.")
            break
        except Exception as e:
            traceback.print_exc()
            # Must reply before next recv
            try:
                socket.send_json({"action": None, "status": f"error: {e}"})
            except Exception:
                pass


if __name__ == "__main__":
    main()
