#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

import cv2
from PIL import Image as PILImage
import numpy as np
import time

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from builtin_interfaces.msg import Time
import os
import sys


MODEL_ID = "/opt/pal/alum/share/hf_models/openvla-7b"

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers."
)

LEFT_ARM_JOINTS = [
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_left_5_joint",
    "arm_left_6_joint",
    "arm_left_7_joint",
]

RIGHT_ARM_JOINTS = [
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
]

# Make motion obvious (radians per step scale)
JOINT_SCALE = np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.5]) * 0.35

# Publish at 2 Hz; each command gives 2 seconds to execute
TIMER_PERIOD = 0.5
TRAJ_TIME = 2.0  # seconds

# Set True to prove your pipeline moves the arm even if VLA outputs tiny numbers
FORCE_SANITY_MOTION = True
SANITY_STEP_RAD = 0.10  # 0.1 rad per tick on joint1 (big + visible)


class VLAArmNode(Node):
    def __init__(self):
        super().__init__("vla_arm_joint_controller")

        # Camera QoS
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.latest_image = None
        self.joint_state = {}  # name -> pos

        self.create_subscription(
            Image,
            "/head_front_camera/color/image_raw",
            self.image_cb,
            cam_qos
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10
        )

        self.left_pub = self.create_publisher(
            JointTrajectory,
            "/arm_left_controller/joint_trajectory",
            10
        )
        self.right_pub = self.create_publisher(
            JointTrajectory,
            "/arm_right_controller/joint_trajectory",
            10
        )

        # OpenVLA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=True
        )

        self.left_goal = None
        self.right_goal = None

        self._last_status_t = 0.0

        # self.create_timer(TIMER_PERIOD, self.control_loop)
        self.create_timer(3.5, self.control_loop)
        self.get_logger().info("VLA joint controller node started")

    def image_cb(self, msg: Image):
        # Your camera is rgb8, so we can convert directly
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            # fallback if driver gives bgr8 sometimes
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.latest_image = PILImage.fromarray(img)

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_state[name] = pos

    def _ready(self):
        img_ok = self.latest_image is not None
        left_ok = all(j in self.joint_state for j in LEFT_ARM_JOINTS)
        right_ok = all(j in self.joint_state for j in RIGHT_ARM_JOINTS)
        return img_ok, left_ok, right_ok

    def control_loop(self):
        img_ok, left_ok, right_ok = self._ready()

        # Periodic status print so you can see if loop is active
        now = time.time()
        if now - self._last_status_t > 1.0:
            self.get_logger().info(
                f"ready: img={img_ok} left_joints={left_ok} right_joints={right_ok}"
            )
            self._last_status_t = now

        if not img_ok or not left_ok:
            return

        # Initialize goals from current state once
        if self.left_goal is None:
            self.left_goal = np.array([self.joint_state[j] for j in LEFT_ARM_JOINTS], dtype=float)
        if right_ok and self.right_goal is None:
            self.right_goal = np.array([self.joint_state[j] for j in RIGHT_ARM_JOINTS], dtype=float)

        # ---- VLA inference ----
        instruction = "pick up one coke can with the left arm"
        prompt = f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction}? ASSISTANT:"

        inputs = self.processor(prompt, self.latest_image, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: (v.to(self.dtype) if v.dtype.is_floating_point else v)
                      for k, v in inputs.items()}

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False
            )

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).reshape(-1)

        # Use first 7 dims as pseudo joint deltas (testing only)
        joint_delta = action[:7] * JOINT_SCALE

        # Optional sanity motion to prove publish path works
        if FORCE_SANITY_MOTION:
            joint_delta = np.zeros(7, dtype=float)
            joint_delta[0] = SANITY_STEP_RAD

        # Accumulate goals (so movement is guaranteed to grow)
        self.left_goal = self.left_goal + joint_delta
        self.publish_arm(self.left_pub, LEFT_ARM_JOINTS, self.left_goal, arm_name="left")

        if right_ok:
            self.right_goal = self.right_goal + joint_delta
            self.publish_arm(self.right_pub, RIGHT_ARM_JOINTS, self.right_goal, arm_name="right")

    def publish_arm(self, pub, joint_names, positions, arm_name="arm"):
        traj = JointTrajectory()
        traj.header.stamp = Time(sec=0, nanosec=0)  # MUST be zero
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions]

        pt.time_from_start.sec = 3
        pt.time_from_start.nanosec = 0

        traj.points = [pt]
        pub.publish(traj)

        self.get_logger().info(
            f"EXEC {arm_name}: j1={pt.positions[0]:.3f}, j2={pt.positions[1]:.3f}"
        )


def main():
    rclpy.init()
    node = VLAArmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
