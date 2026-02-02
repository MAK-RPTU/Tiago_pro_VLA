#!/usr/bin/env python3
# OpenVLA formula: delta_pose (dx,dy,dz + R from droll,dpitch,dyaw), target_pose = current_pose @ delta_pose.
# Publishes /vla/target_pose (PoseStamped). Executor should use setPoseTarget(target_pose) and move().
# When running in simulation: ros2 launch vla_startup vla_frame_debug.launch.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge
from PIL import Image as PILImage

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

import tf2_ros
from tf2_ros import TransformException

from rclpy.time import Time


# ---------------- CONFIG ----------------
MODEL_ID = "/opt/pal/alum/share/hf_models/openvla-7b"

CAMERA_TOPIC = "/head_front_camera/color/image_raw"
CAMERA_FRAME = "head_front_camera_color_optical_frame"
BASE_FRAME = "base_footprint"   # match MoveIt planning frame
EE_LINK = "arm_left_7_link"     # end-effector link for left arm (Tiago Pro)

POS_SCALE_M = 0.30               # scale normalized [-1,1] → meters for dx,dy,dz
ROT_SCALE_RAD = 0.15             # scale normalized [-1,1] → radians for droll,dpitch,dyaw
TIMER_PERIOD = 2.0               # seconds
# ----------------------------------------


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build 3x3 rotation matrix from roll, pitch, yaw (radians). R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return R


class VLAFrameDebugNode(Node):
    def __init__(self):
        super().__init__("vla_frame_debug")

        self.bridge = CvBridge()
        self.latest_image = None

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_cb,
            cam_qos,
        )

        self.cam_pub = self.create_publisher(PointStamped, "/vla/camera_point", 10)
        self.base_pub = self.create_publisher(PointStamped, "/vla/base_point", 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, "/vla/target_pose", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            attn_implementation="eager",
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.create_timer(TIMER_PERIOD, self.step)
        self.get_logger().info("✅ VLA frame debug node started (OpenVLA: target_pose = current_pose @ delta_pose)")

    def image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.latest_image = PILImage.fromarray(img)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def step(self):
        self.get_logger().info("step() running")

        if self.latest_image is None:
            self.get_logger().warn("No image received yet")
            return

        prompt = "reach the object in front of the robot"

        inputs = self.processor(prompt, self.latest_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
            )

        action = np.asarray(action).reshape(-1)

        # OpenVLA: action[:3] = dx, dy, dz (position delta); action[3:6] = droll, dpitch, dyaw (rotation delta)
        dx, dy, dz = action[0] * POS_SCALE_M, action[1] * POS_SCALE_M, action[2] * POS_SCALE_M
        droll = action[3] * ROT_SCALE_RAD if len(action) > 3 else 0.0
        dpitch = action[4] * ROT_SCALE_RAD if len(action) > 4 else 0.0
        dyaw = action[5] * ROT_SCALE_RAD if len(action) > 5 else 0.0

        a3 = float(action[3]) if len(action) > 3 else 0.0
        a4 = float(action[4]) if len(action) > 4 else 0.0
        a5 = float(action[5]) if len(action) > 5 else 0.0
        self.get_logger().info(
            f"[OpenVLA raw] pos=({action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}) "
            f"rpy=({a3:.4f}, {a4:.4f}, {a5:.4f})"
        )

        lookup_stamp = Time(seconds=0, nanoseconds=0).to_msg()

        try:
            # Current end-effector pose in base frame (from TF)
            trans = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                EE_LINK,
                lookup_stamp,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed (current EE pose): {e}")
            return

        # current_pose 4x4 (base frame)
        t = trans.transform.translation
        r = trans.transform.rotation
        R_cur = quaternion_to_rotation_matrix(r.x, r.y, r.z, r.w)
        current_pose = np.eye(4, dtype=np.float64)
        current_pose[:3, :3] = R_cur
        current_pose[0, 3], current_pose[1, 3], current_pose[2, 3] = t.x, t.y, t.z

        # delta_pose 4x4 (in end-effector frame): translation + rotation
        R_delta = euler_to_rotation_matrix(droll, dpitch, dyaw)
        delta_pose = np.eye(4, dtype=np.float64)
        delta_pose[:3, :3] = R_delta
        delta_pose[0, 3], delta_pose[1, 3], delta_pose[2, 3] = dx, dy, dz

        # OpenVLA: target_pose = current_pose @ delta_pose
        target_pose = current_pose @ delta_pose

        # PoseStamped for /vla/target_pose
        target_msg = PoseStamped()
        target_msg.header.stamp = lookup_stamp
        target_msg.header.frame_id = BASE_FRAME
        target_msg.pose.position.x = float(target_pose[0, 3])
        target_msg.pose.position.y = float(target_pose[1, 3])
        target_msg.pose.position.z = float(target_pose[2, 3])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(target_pose[:3, :3])
        target_msg.pose.orientation.x = qx
        target_msg.pose.orientation.y = qy
        target_msg.pose.orientation.z = qz
        target_msg.pose.orientation.w = qw

        self.target_pose_pub.publish(target_msg)

        # Legacy: publish target position as PointStamped for debugging / backward compat
        base_pt = PointStamped()
        base_pt.header = target_msg.header
        base_pt.point = target_msg.pose.position
        self.base_pub.publish(base_pt)

        self.get_logger().info(
            f"[target_pose] frame={BASE_FRAME} pos=("
            f"{target_msg.pose.position.x:.4f}, {target_msg.pose.position.y:.4f}, "
            f"{target_msg.pose.position.z:.4f}) quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
        )


def main():
    rclpy.init()
    node = VLAFrameDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
