from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp

logger_mp = logging_mp.get_logger(__name__)

FTP_Num_Motors = 6  # 6 finger joints per hand

# FTP hand DDS topics
kTopicInspireFTPLeftCommand  = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState    = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState   = "rt/inspire_hand/state/r"


class Inspire_Controller_FTP:
    """Eval-side controller for the Inspire FTP dexterous hand.

    Subscribes to rt/inspire_hand/state/l|r and publishes to rt/inspire_hand/ctrl/l|r.
    The policy action is expected to be in [0, 1] per joint; it is scaled to [0, 1000]
    for the hardware angle command.
    """

    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        import inspire_sdkpy.inspire_dds as inspire_dds
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
        self._inspire_dds = inspire_dds
        self._inspire_hand_default = inspire_hand_default

        logger_mp.info("Initialize Inspire_Controller_FTP...")
        self.fps = fps
        self.simulation_mode = simulation_mode

        # ChannelFactory already initialized by arm controller — do not call again

        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array  = Array("d", FTP_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", FTP_Num_Motors, lock=True)
        self.left_state_received  = False
        self.right_state_received = False

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        wait_count = 0
        while not (self.left_state_received and self.right_state_received):
            if wait_count % 100 == 0:
                logger_mp.info(
                    f"[Inspire_Controller_FTP] Waiting for hand states (L:{self.left_state_received} R:{self.right_state_received})..."
                )
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning("[Inspire_Controller_FTP] Timeout waiting for hand states. Proceeding anyway.")
                break
        logger_mp.info("[Inspire_Controller_FTP] Hand states ready.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_msg = self.LeftHandState_subscriber.Read()
            if left_msg is not None and hasattr(left_msg, "angle_act") and len(left_msg.angle_act) == FTP_Num_Motors:
                with self.left_hand_state_array.get_lock():
                    for i in range(FTP_Num_Motors):
                        self.left_hand_state_array[i] = left_msg.angle_act[i] / 1000.0
                self.left_state_received = True

            right_msg = self.RightHandState_subscriber.Read()
            if right_msg is not None and hasattr(right_msg, "angle_act") and len(right_msg.angle_act) == FTP_Num_Motors:
                with self.right_hand_state_array.get_lock():
                    for i in range(FTP_Num_Motors):
                        self.right_hand_state_array[i] = right_msg.angle_act[i] / 1000.0
                self.right_state_received = True

            time.sleep(0.002)

    def _send_hand_command(self, left_scaled, right_scaled):
        left_cmd = self._inspire_hand_default.get_inspire_hand_ctrl()
        left_cmd.angle_set = left_scaled
        left_cmd.mode = 0b0001
        self.LeftHandCmd_publisher.Write(left_cmd)

        right_cmd = self._inspire_hand_default.get_inspire_hand_ctrl()
        right_cmd.angle_set = right_scaled
        right_cmd.mode = 0b0001
        self.RightHandCmd_publisher.Write(right_cmd)

    def control_process(
        self,
        left_gripper_value,
        right_gripper_value,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        """
        left_gripper_value / right_gripper_value: multiprocessing.Value('d')
            Policy outputs a single [0,1] gripper value per hand.
            0 = fully closed, 1 = fully open.
            All 6 FTP finger joints receive the same scaled command.
        dual_hand_state_array: Array of size 2 — [left_gripper_state, right_gripper_state]
        """
        self.running = True
        left_val  = 1.0  # default open
        right_val = 1.0

        try:
            while self.running:
                start_time = time.time()

                # Read policy action (single gripper value per hand)
                with left_gripper_value.get_lock():
                    left_val = float(left_gripper_value.value)
                with right_gripper_value.get_lock():
                    right_val = float(right_gripper_value.value)

                # State: mean of 6 finger positions → single [0,1] value per hand
                with left_hand_state_array.get_lock():
                    left_state = float(np.mean(np.array(left_hand_state_array[:])))
                with right_hand_state_array.get_lock():
                    right_state = float(np.mean(np.array(right_hand_state_array[:])))

                if dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = [left_state, right_state]
                        dual_hand_action_array[:] = [left_val, right_val]

                # Apply single gripper value to all 6 fingers
                scaled = int(np.clip(left_val * 1000, 0, 1000))
                scaled_left = [scaled] * FTP_Num_Motors
                scaled = int(np.clip(right_val * 1000, 0, 1000))
                scaled_right = [scaled] * FTP_Num_Motors
                self._send_hand_command(scaled_left, scaled_right)

                time.sleep(max(0, (1 / self.fps) - (time.time() - start_time)))
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")

Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"


class Inspire_Controller:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0, "enP8p1s0")

        # initialize handcmd publisher and handstate subscriber
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array):  # any(self.left_hand_state_array) and
                break
            time.sleep(0.01)
            logger_mp.warning("[Inspire_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Inspire_Controller] Subscribe dds ok.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True

        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))
        ]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()

                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_mat = np.array(left_hand_array[:]).copy()
                with right_hand_array.get_lock():
                    right_hand_mat = np.array(right_hand_array[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                action_data = np.concatenate((left_hand_mat, right_hand_mat))
                if dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        left_q_target = left_hand_mat
                        right_q_target = right_hand_mat

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")


# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5


class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
