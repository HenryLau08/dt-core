#!/usr/bin/env python3

import os
import yaml
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped
from sensor_msgs.msg import CompressedImage, CameraInfo

from solution.model import MLModel


class ObjectDetectionNode(DTROS):
    """
    Runs the ONNX duckie-detection model on incoming camera frames and
    publishes a BoolStamped on ~object_detected (True = stop, False = clear).

    Detection logic:
      - Run ONNX detector on the image
      - For each high-confidence detection, check if the bounding box bottom
        edge is below a threshold fraction of the image height. If so, the
        duckie is "close enough" to stop for.
      - This avoids any ground-projection import issues while still giving
        a reasonable distance proxy: objects lower in the frame are closer.

    The ~stop_line_threshold param (0.0-1.0) is the fraction of image height
    below which a detection counts as "too close". Default 0.7 means the
    bottom of the bounding box must be in the lower 30% of the image.

    Subscribers:
        ~image/compressed  (sensor_msgs/CompressedImage)
        ~camera_info       (sensor_msgs/CameraInfo)   [for image dimensions]

    Publishers:
        ~object_detected   (duckietown_msgs/BoolStamped)
    """

    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        self.bridge = CvBridge()
        self._img_height = None  # set from first CameraInfo or first image

        # ── Parameters ────────────────────────────────────────────────────────
        self._conf_threshold    = DTParam(
            "~conf_threshold",    param_type=ParamType.FLOAT, default=0.5)
        self._stop_line_thresh  = DTParam(
            "~stop_line_threshold", param_type=ParamType.FLOAT, default=0.70)
        # stop_line_threshold: fraction of image height.
        # Detection whose bbox bottom-edge is BELOW this line triggers a stop.
        # 0.70 = bottom 30% of image → duckie is close.

        # ── Model ─────────────────────────────────────────────────────────────
        self.log("Loading ONNX model ...")
        try:
            self.model = MLModel()
            self.log(f"ONNX model loaded. Input size: "
                     f"{self.model.net_h}x{self.model.net_w}")
        except FileNotFoundError as e:
            self.logerr(str(e))
            raise

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_camera_info = rospy.Subscriber(
            "~camera_info", CameraInfo, self._cb_camera_info, queue_size=1
        )
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self._cb_image,
            queue_size=1, buff_size=2 ** 24
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_object_detected = rospy.Publisher(
            "~object_detected", BoolStamped, queue_size=1
        )

        self.log("ObjectDetectionNode initialised.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo):
        if self._img_height is None:
            self._img_height = msg.height
            self.log(f"Camera info received: {msg.width}x{msg.height}")

    def _cb_image(self, msg: CompressedImage):
        # Decode
        try:
            img_bgr = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception as e:
            self.logerr(f"Image decode error: {e}")
            return

        orig_h, orig_w = img_bgr.shape[:2]

        # Cache image height from first frame if camera_info hasn't arrived yet
        if self._img_height is None:
            self._img_height = orig_h

        # Resize to ONNX input size
        net_h, net_w = self.model.net_h, self.model.net_w
        if orig_h != net_h or orig_w != net_w:
            img_resized = cv2.resize(img_bgr, (net_w, net_h))
        else:
            img_resized = img_bgr

        # Run ONNX detector
        try:
            detections = self.model._run_detector(img_resized)
        except Exception as e:
            self.logerr(f"Inference error: {e}")
            return

        # Decide stop based on bounding box position
        stop = self._should_stop_by_bbox(detections, net_h, net_w)

        # Publish
        out = BoolStamped()
        out.header = msg.header
        out.data = bool(stop)
        self.pub_object_detected.publish(out)

    def _should_stop_by_bbox(self, detections: np.ndarray,
                              net_h: int, net_w: int) -> bool:
        """
        Returns True if any high-confidence detection has its bounding box
        bottom edge below stop_line_threshold * net_h.

        Coordinates in the ONNX output are in pixels relative to the
        (net_h x net_w) input image.
        """
        threshold_y = self._stop_line_thresh.value * net_h

        for x1, y1, x2, y2, score, cls in detections:
            if score < self._conf_threshold.value:
                continue
            # y2 is the bottom of the bounding box
            if y2 > threshold_y:
                self.log(
                    f"Duckie detected! score={score:.2f} "
                    f"bbox_bottom={y2:.1f} > threshold={threshold_y:.1f}"
                )
                return True
        return False


if __name__ == "__main__":
    node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
