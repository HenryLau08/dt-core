#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, WheelsCmdStamped
from sensor_msgs.msg import CompressedImage, CameraInfo

from dt_computer_vision.camera.types import CameraModel
from dt_computer_vision.ground_projection.types import GroundProjectionGeometry

from solution.model import MLModel


class ObjectDetectionNode(DTROS):
    """
    Runs the ONNX duckie-detection model on incoming camera frames and
    publishes wheel commands (stop / forward) and a detection flag.

    This node is modelled after the obstacle_detection package in dt-core.

    Configuration (config/object_detection_node/default.yaml):
        ~conf_threshold  (float): Minimum detection confidence.
        ~stop_distance   (float): Distance in metres below which the bot stops.
        ~forward_pwm     (float): PWM value used for straight-line driving.

    Subscribers:
        ~image/compressed  (sensor_msgs/CompressedImage): Camera feed.
        ~camera_info       (sensor_msgs/CameraInfo):      Camera intrinsics.

    Publishers:
        ~wheels_cmd        (duckietown_msgs/WheelsCmdStamped): Motor commands.
        ~object_detected   (duckietown_msgs/BoolStamped):      Detection flag.
    """

    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        self.bridge = CvBridge()
        self._camera_info_received = False

        # ── Parameters (loaded from YAML, overridable at launch) ─────────────
        self._conf_threshold = DTParam("~conf_threshold", param_type=ParamType.FLOAT, default=0.5)
        self._stop_distance  = DTParam("~stop_distance",  param_type=ParamType.FLOAT, default=0.5)
        self._forward_pwm    = DTParam("~forward_pwm",    param_type=ParamType.FLOAT, default=0.4)

        # ── Model ─────────────────────────────────────────────────────────────
        self.log("Loading ONNX model …")
        try:
            self.model = MLModel()
            self.log("ONNX model loaded successfully.")
        except FileNotFoundError as e:
            self.logerr(str(e))
            raise

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_camera_info = rospy.Subscriber(
            "~camera_info", CameraInfo, self.cb_camera_info, queue_size=1
        )
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.cb_image,
            queue_size=1, buff_size=2 ** 24
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_wheels_cmd = rospy.Publisher(
            "~wheels_cmd", WheelsCmdStamped, queue_size=1
        )
        self.pub_object_detected = rospy.Publisher(
            "~object_detected", BoolStamped, queue_size=1
        )

        self.log("ObjectDetectionNode ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_camera_info(self, msg: CameraInfo):
        """Build the ground projector once from the first CameraInfo message."""
        if self._camera_info_received:
            return
        try:
            camera_model     = CameraModel.from_camera_info(msg)
            ground_projector = GroundProjectionGeometry(camera=camera_model)
            self.model.set_ground_projector(ground_projector)
            self._camera_info_received = True
            self.log("Ground projector initialised.")
        except Exception as e:
            self.logerr(f"Failed to initialise ground projector: {e}")

    def cb_image(self, msg: CompressedImage):
        """Decode image → run model → publish wheel command + detection flag."""
        if not self._camera_info_received:
            self.logwarn_throttle(5.0, "Waiting for camera_info before processing images …")
            return

        # Decode
        try:
            img_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.logerr(f"Image decode error: {e}")
            return

        # Resize to model input size if needed
        net_h, net_w = self.model.net_h, self.model.net_w
        h, w = img_bgr.shape[:2]
        if h != net_h or w != net_w:
            img_bgr = cv2.resize(img_bgr, (net_w, net_h))

        # Inference
        try:
            pwm, detections = self.model.get_wheel_velocities_from_image(img_bgr)
        except Exception as e:
            self.logerr(f"Model inference error: {e}")
            return

        # Publish wheel command
        wheels_cmd           = WheelsCmdStamped()
        wheels_cmd.header    = msg.header
        wheels_cmd.vel_left  = float(pwm.left)
        wheels_cmd.vel_right = float(pwm.right)
        self.pub_wheels_cmd.publish(wheels_cmd)

        # Publish detection flag
        object_detected      = BoolStamped()
        object_detected.header = msg.header
        object_detected.data   = detections is not None and len(detections) > 0
        self.pub_object_detected.publish(object_detected)


if __name__ == "__main__":
    node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
