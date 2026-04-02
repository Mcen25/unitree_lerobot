"""
OpenVLA inference server with V-GPS multi-sample support.

Supports three modes:
  Standard OpenVLA (default): token-based autoregressive prediction, returns single action.
  OpenVLA-OFT: action head (MLP) regression, returns full action chunk.
  V-GPS: sample N actions with temperature for client-side Cal-QL value scoring.

For V-GPS, the client sends {"image": ..., "task": ..., "num_samples": N, "sample_temperature": T}.
The server returns {"actions": [[...], ...], "status": "ok"} with N 7-DOF action arrays.

Pass --no-action-head to force standard mode even if the checkpoint contains an action head.

The checkpoint directory must contain:
  - model-*.safetensors / model.safetensors   merged model weights (OFT checkpoints)
  - dataset_statistics.json                   action normalization statistics
  - [OFT only] action_head--*_checkpoint.pt   L1RegressionActionHead weights
  - [OFT only] modeling_prismatic.py          OFT model code (trust_remote_code)

Usage (Jetson Thor):
    conda activate <env>
    python openvla_server_v_gps.py \\
        --checkpoint /path/to/openvla-7b+pick_up_bottle_1+... \\
        --no-action-head \\
        --port 5555

Notes for Jetson Thor (aarch64):
  - flash_attention_2 is NOT used — eager attention is used instead
  - float16 weights are recommended (stable on all Jetson variants)
  - 64 GB unified memory is sufficient for openvla-7b in float16 (~14 GB)
"""

import argparse
import base64
import json
import os
import sys
import traceback
from io import BytesIO

import numpy as np
import zmq
from PIL import Image


# ---------------------------------------------------------------------------
# Minimal prismatic stubs
#
# The checkpoint's modeling_prismatic.py imports from prismatic.training.train_utils
# and prismatic.vla.constants.  The full openvla-oft prismatic package pulls in
# torch.distributed.fsdp which is broken on this PyTorch build.  We inject
# lightweight stubs into sys.modules before HuggingFace's trust_remote_code
# mechanism runs check_imports(), so it never touches the real package.
# ---------------------------------------------------------------------------

def _inject_prismatic_stubs(num_actions_chunk: int = 8) -> None:
    """Populate sys.modules with the minimal prismatic shims that modeling_prismatic.py needs."""
    import types
    from enum import Enum

    # Skip if real package is already loaded and working.
    if "prismatic" in sys.modules and not getattr(sys.modules["prismatic"], "_is_stub", False):
        return

    # --- prismatic.vla.constants ---
    class NormalizationType(str, Enum):
        NORMAL = "normal"
        BOUNDS = "bounds"
        BOUNDS_Q99 = "bounds_q99"

    constants_mod = types.ModuleType("prismatic.vla.constants")
    constants_mod.ACTION_DIM = 7
    constants_mod.ACTION_TOKEN_BEGIN_IDX = 31743
    constants_mod.IGNORE_INDEX = -100
    constants_mod.STOP_INDEX = 2
    constants_mod.NUM_ACTIONS_CHUNK = num_actions_chunk
    constants_mod.NormalizationType = NormalizationType
    constants_mod.ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99

    # --- prismatic.training.train_utils ---
    # These two functions are the only things modeling_prismatic.py uses from the package.
    def get_current_action_mask(token_ids):
        import torch
        newline_positions = token_ids != -100           # IGNORE_INDEX
        cumsum = torch.cumsum(newline_positions, dim=1)
        mask = (1 <= cumsum) & (cumsum <= 7)            # ACTION_DIM
        return (token_ids > 31743) * mask               # ACTION_TOKEN_BEGIN_IDX

    def get_next_actions_mask(token_ids):
        import torch
        newline_positions = token_ids != -100
        cumsum = torch.cumsum(newline_positions, dim=1)
        mask = cumsum > 7
        return (token_ids > 31743) * mask

    train_utils_mod = types.ModuleType("prismatic.training.train_utils")
    train_utils_mod.get_current_action_mask = get_current_action_mask
    train_utils_mod.get_next_actions_mask = get_next_actions_mask

    # Build the package hierarchy so attribute access works too.
    prismatic_mod  = types.ModuleType("prismatic");  prismatic_mod._is_stub = True
    vla_mod        = types.ModuleType("prismatic.vla")
    training_mod   = types.ModuleType("prismatic.training")
    models_mod     = types.ModuleType("prismatic.models")

    prismatic_mod.vla      = vla_mod
    prismatic_mod.training = training_mod
    prismatic_mod.models   = models_mod
    vla_mod.constants      = constants_mod
    training_mod.train_utils = train_utils_mod

    sys.modules.update({
        "prismatic":                       prismatic_mod,
        "prismatic.vla":                   vla_mod,
        "prismatic.vla.constants":         constants_mod,
        "prismatic.training":              training_mod,
        "prismatic.training.train_utils":  train_utils_mod,
        "prismatic.models":                models_mod,
    })


# ---------------------------------------------------------------------------
# OpenVLA prompt template
# ---------------------------------------------------------------------------

def make_prompt(task: str) -> str:
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA inference server (standard or OFT)")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the checkpoint directory (merged safetensors + dataset_statistics.json). "
             "Omit to run the base model.",
    )
    p.add_argument(
        "--no-action-head",
        action="store_true",
        help="Skip loading the OFT action head and use standard token-based prediction instead. "
             "Use this for standard OpenVLA (non-OFT) fine-tunes.",
    )
    p.add_argument(
        "--base-model",
        default="openvla/openvla-7b",
        help="Base model: HuggingFace ID or local snapshot path. "
             "Only used when the checkpoint has no merged model weights.",
    )
    p.add_argument("--port", type=int, default=5555, help="ZMQ REP port")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype (float16 recommended for Jetson aarch64).",
    )
    p.add_argument(
        "--num-actions-chunk",
        type=int,
        default=8,
        help="Action chunk size used during fine-tuning (default: 8, matching the "
             "LIBERO/default constant used by openvla-oft when no platform keyword "
             "appears in the training command).",
    )
    p.add_argument(
        "--open-loop-steps",
        type=int,
        default=1,
        help="Steps to execute open-loop from each predicted chunk before re-running "
             "inference. 1 = closed-loop (infer every step, recommended). "
             "Set equal to --num-actions-chunk for fully open-loop execution.",
    )
    p.add_argument(
        "--unnorm-key",
        default="bridge_orig",
        help="Action unnormalization key when running without a checkpoint.",
    )
    # Legacy LoRA-merge options (ignored when checkpoint has merged weights)
    p.add_argument("--merged-cache", default=None,
                   help="[Legacy] Directory to cache a LoRA-merged model.")
    p.add_argument("--clear-cache", action="store_true",
                   help="[Legacy] Delete merged model cache and re-merge.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _has_merged_weights(checkpoint_path: str | None) -> bool:
    if not checkpoint_path:
        return False
    return (
        os.path.isfile(os.path.join(checkpoint_path, "model.safetensors.index.json"))
        or os.path.isfile(os.path.join(checkpoint_path, "model.safetensors"))
    )


def load_model_and_processor(
    checkpoint_path: str | None,
    base_model: str,
    dtype_str: str,
    num_actions_chunk: int = 8,
    merged_cache: str | None = None,
    clear_cache: bool = False,
):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[dtype_str]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server] Device: {device}")

    # ------------------------------------------------------------------
    # OFT path: checkpoint already contains merged model weights.
    # Inject prismatic stubs BEFORE from_pretrained so HuggingFace's
    # check_imports() resolves prismatic.* without loading the full package.
    # ------------------------------------------------------------------
    if _has_merged_weights(checkpoint_path):
        _inject_prismatic_stubs(num_actions_chunk)

        print(f"[server] Loading processor from {checkpoint_path} ...")
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

        print(f"[server] Loading merged OFT model from {checkpoint_path} ...")
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
        with open(stats_path) as f:
            custom_stats = json.load(f)
        model.norm_stats.update(custom_stats)
        print(f"[server] Injected norm_stats keys: {list(custom_stats.keys())}")

    # ------------------------------------------------------------------
    # Legacy path: no merged weights → load base model and merge LoRA.
    # ------------------------------------------------------------------
    else:
        processor_src = checkpoint_path if checkpoint_path else base_model
        print(f"[server] Loading processor from {processor_src} ...")
        processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)

        print(f"[server] Loading base model from '{base_model}' ...")
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if checkpoint_path:
            import shutil

            stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
            with open(stats_path) as f:
                custom_stats = json.load(f)

            if clear_cache and merged_cache and os.path.isdir(merged_cache):
                print(f"[server] --clear-cache: removing {merged_cache} ...")
                shutil.rmtree(merged_cache)

            if merged_cache and os.path.isdir(merged_cache):
                print(f"[server] Loading merged model from cache: {merged_cache} ...")
                model = AutoModelForVision2Seq.from_pretrained(
                    merged_cache, torch_dtype=dtype, low_cpu_mem_usage=True,
                    trust_remote_code=True, attn_implementation="eager",
                )
            else:
                from peft import PeftModel
                lora_path = os.path.join(checkpoint_path, "lora_adapter")
                print(f"[server] Merging LoRA adapter from {lora_path} ...")
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                if merged_cache:
                    print(f"[server] Saving merged model to cache: {merged_cache} ...")
                    model.save_pretrained(merged_cache)
                    processor.save_pretrained(merged_cache)

            model.norm_stats.update(custom_stats)
            print(f"[server] Injected norm_stats keys: {list(custom_stats.keys())}")
        else:
            print("[server] No checkpoint — running base model without LoRA.")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[server] Model ready — {n_params:.1f}B params, dtype={dtype}")
    return model, processor, device


# ---------------------------------------------------------------------------
# Action head
# ---------------------------------------------------------------------------

def load_action_head(checkpoint_path: str, device: str, dtype_str: str):
    """Load L1RegressionActionHead from action_head--*_checkpoint.pt."""
    import glob
    import torch
    import torch.nn as nn

    pattern = os.path.join(checkpoint_path, "action_head--*_checkpoint.pt")
    matches = glob.glob(pattern)
    if not matches:
        print("[server] No action head checkpoint found — using token-based prediction.")
        return None

    action_head_path = matches[0]
    print(f"[server] Loading action head from {action_head_path} ...")

    # Self-contained implementation matching openvla-oft exactly.
    class _MLPResNetBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

        def forward(self, x):
            return x + self.ffn(x)

    class _MLPResNet(nn.Module):
        def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer_norm1 = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.mlp_resnet_blocks = nn.ModuleList(
                [_MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
            )
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.layer_norm1(x)
            x = self.fc1(x)
            x = self.relu(x)
            for block in self.mlp_resnet_blocks:
                x = block(x)
            x = self.layer_norm2(x)
            return self.fc2(x)

    class _L1RegressionActionHead(nn.Module):
        _ACTION_DIM = 7
        _LLM_DIM    = 4096

        def __init__(self):
            super().__init__()
            self.model = _MLPResNet(
                num_blocks=2,
                input_dim=self._LLM_DIM * self._ACTION_DIM,  # 28672
                hidden_dim=self._LLM_DIM,
                output_dim=self._ACTION_DIM,
            )

        def predict_action(self, actions_hidden_states):
            # actions_hidden_states: (B, num_chunks * ACTION_DIM, LLM_DIM)
            B, n_tokens, llm_dim = actions_hidden_states.shape
            num_chunks = n_tokens // self._ACTION_DIM
            # Reshape so each chunk has ACTION_DIM consecutive hidden states concatenated
            rearranged = actions_hidden_states.reshape(B, num_chunks, self._ACTION_DIM * llm_dim)
            return self.model(rearranged)  # (B, num_chunks, ACTION_DIM)

    action_head = _L1RegressionActionHead()

    # Load weights, stripping DDP "module." prefix if present.
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    raw_sd = torch.load(action_head_path, map_location="cpu", weights_only=True)
    clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
    action_head.load_state_dict(clean_sd)
    action_head = action_head.to(device=device, dtype=dtype_map[dtype_str])
    action_head.eval()
    print("[server] Action head ready.")
    return action_head


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _prepare_inputs(model, processor, image, task, device):
    """Prepare model inputs once, reusable for multiple sampling passes."""
    prompt = make_prompt(task)
    inputs = processor(prompt, image)
    inputs = {
        k: (
            v.to(device, dtype=model.dtype) if (hasattr(v, "to") and v.is_floating_point())
            else v.to(device) if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }
    return {k: v for k, v in inputs.items() if k != "attention_mask"}


def run_inference(
    model,
    processor,
    action_head,
    image: Image.Image,
    task: str,
    unnorm_key: str,
    device: str,
) -> list[list[float]]:
    """
    Run one forward pass and return the predicted action chunk.

    OFT path (action_head provided):
        Calls the OFT predict_action() which appends NUM_ACTIONS_CHUNK*7 placeholder
        tokens, zeros their embeddings, does a single non-autoregressive forward pass,
        extracts hidden states at the action positions, and passes them through the MLP
        action head. Returns a list of num_actions_chunk 7-element action arrays.

    Legacy path (no action_head):
        Autoregressively generates 7 action tokens and decodes them via vocabulary
        unnormalization. Returns a list containing one 7-element action array.
    """
    import torch

    prompt = make_prompt(task)
    inputs = processor(prompt, image)
    inputs = {
        k: (
            v.to(device, dtype=model.dtype) if (hasattr(v, "to") and v.is_floating_point())
            else v.to(device) if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }

    with torch.no_grad():
        if action_head is not None:
            # OFT single-pass inference.  The model's predict_action (from the
            # checkpoint's OFT modeling_prismatic.py) signature is:
            #   predict_action(input_ids, unnorm_key, action_head, **kwargs)
            # where **kwargs carries pixel_values and attention_mask.
            result = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                action_head=action_head,
                do_sample=False,
            )
            actions = result[0]  # (num_actions_chunk, 7) numpy array
        else:
            # Standard token path: drop attention_mask before calling generate().
            # The text-only attention_mask has shape (1, text_len), but after visual
            # token injection the sequence length is text_len+1. HuggingFace's generate()
            # builds a causal mask from the original attention_mask size, causing a size
            # mismatch in LLaMA attention. Dropping it lets the model use a full causal
            # mask (all ones), which is correct for a single-sample batch.
            inputs_no_mask = {k: v for k, v in inputs.items() if k != "attention_mask"}
            actions = model.predict_action(**inputs_no_mask, unnorm_key=unnorm_key, do_sample=False)
            if hasattr(actions, "cpu"):
                actions = actions.cpu().numpy()
            actions = np.atleast_2d(actions)  # (1, 7)

    return [[float(x) for x in step] for step in np.atleast_2d(actions)]


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model, processor, device = load_model_and_processor(
        args.checkpoint,
        args.base_model,
        args.dtype,
        num_actions_chunk=args.num_actions_chunk,
        merged_cache=args.merged_cache,
        clear_cache=args.clear_cache,
    )

    action_head = None
    if args.checkpoint and not args.no_action_head:
        action_head = load_action_head(args.checkpoint, device, args.dtype)
    elif args.no_action_head:
        print("[server] --no-action-head: skipping action head, using token-based prediction.")

    if args.checkpoint:
        stats_path = os.path.join(args.checkpoint, "dataset_statistics.json")
        with open(stats_path) as f:
            dataset_stats = json.load(f)
        unnorm_key = list(dataset_stats.keys())[0]
    else:
        unnorm_key = args.unnorm_key

    print(f"[server] unnorm_key = '{unnorm_key}'")
    if action_head is not None:
        print(f"[server] OFT mode | num_actions_chunk={args.num_actions_chunk}")
    else:
        print("[server] Standard OpenVLA mode (token-based prediction).")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[server] Listening on port {args.port} ...")

    # Chunk management and temporal ensembling are now handled client-side.
    # The server always runs a fresh inference and returns the full chunk.

    step = 0
    while True:
        try:
            msg = socket.recv_json()
            task = msg.get("task", unnorm_key)
            image = Image.open(BytesIO(base64.b64decode(msg["image"]))).convert("RGB")
            num_samples = int(msg.get("num_samples", 1))
            sample_temperature = float(msg.get("sample_temperature", 1.5))

            if action_head is not None:
                # OFT: return full chunk for client-side temporal ensembling
                chunk = run_inference(model, processor, action_head, image, task, unnorm_key, device)
                if step % 20 == 0:
                    print(f"[server] step={step} | action={np.round(chunk[0], 4)}")
                socket.send_json({"actions": chunk, "status": "ok"})
            elif num_samples > 1:
                # V-GPS: sample N actions with temperature for client-side value scoring
                inputs_no_mask = _prepare_inputs(model, processor, image, task, device)
                sampled = []
                import torch
                with torch.no_grad():
                    for _ in range(num_samples):
                        a = model.predict_action(**inputs_no_mask, unnorm_key=unnorm_key,
                                                 do_sample=True, temperature=sample_temperature)
                        if hasattr(a, "cpu"):
                            a = a.cpu().numpy()
                        sampled.append([float(x) for x in np.atleast_1d(a).flatten()[:7]])
                if step % 20 == 0:
                    print(f"[server] step={step} | V-GPS {num_samples} samples | first={np.round(sampled[0], 4)}")
                socket.send_json({"actions": sampled, "status": "ok"})
            else:
                # Standard OpenVLA: return single action
                chunk = run_inference(model, processor, action_head, image, task, unnorm_key, device)
                if step % 20 == 0:
                    print(f"[server] step={step} | action={np.round(chunk[0], 4)}")
                socket.send_json({"action": chunk[0], "status": "ok"})

        except KeyboardInterrupt:
            print("[server] Shutting down.")
            break
        except Exception as e:
            traceback.print_exc()
            try:
                socket.send_json({"actions": None, "status": f"error: {e}"})
            except Exception:
                pass


if __name__ == "__main__":
    main()
