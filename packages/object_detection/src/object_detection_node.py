#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, WheelsCmdStamped, FSMState
from sensor_msgs.msg import CompressedImage, CameraInfo

from solution.model import MLModel


_ST_WATCHING = "WATCHING"
_ST_STOPPED  = "STOPPED"
_ST_AVOIDING = "AVOIDING"

# Colours for debug overlay (BGR)
_COL_CLOSE  = (0,   0,   255)   # red   — duckie triggers stop
_COL_FAR    = (0,   165, 255)   # orange — duckie detected but still far
_COL_LINE   = (0,   255, 255)   # yellow — stop threshold line
_COL_TEXT   = (255, 255, 255)   # white text


class ObjectDetectionNode(DTROS):
    """
    ONNX duckie detection + patience-based avoidance node.

    Publishers:
        ~object_detected          (BoolStamped)
        ~avoidance_done           (BoolStamped)
        ~wheels_cmd               (WheelsCmdStamped)
        ~debug/image/compressed   (CompressedImage)  ← viewable in image_viewer
    """

    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        self.bridge     = CvBridge()
        self._img_h     = None
        self._img_w     = None
        self._fsm_state = None

        # ── Parameters ────────────────────────────────────────────────────────
        self._conf_threshold    = DTParam("~conf_threshold",
                                          param_type=ParamType.FLOAT, default=0.5)
        self._stop_thresh       = DTParam("~stop_line_threshold",
                                          param_type=ParamType.FLOAT, default=0.65)
        self._patience_secs     = DTParam("~patience_secs",
                                          param_type=ParamType.FLOAT, default=3.0)
        self._avoid_steer_secs  = DTParam("~avoid_steer_secs",
                                          param_type=ParamType.FLOAT, default=1.2)
        self._avoid_fwd_secs    = DTParam("~avoid_fwd_secs",
                                          param_type=ParamType.FLOAT, default=1.5)
        self._avoid_return_secs = DTParam("~avoid_return_secs",
                                          param_type=ParamType.FLOAT, default=1.0)
        self._avoid_pwm_fast    = DTParam("~avoid_pwm_fast",
                                          param_type=ParamType.FLOAT, default=0.35)
        self._avoid_pwm_slow    = DTParam("~avoid_pwm_slow",
                                          param_type=ParamType.FLOAT, default=0.15)
        self._avoid_pwm_fwd     = DTParam("~avoid_pwm_fwd",
                                          param_type=ParamType.FLOAT, default=0.30)

        # ── Internal avoidance state ───────────────────────────────────────────
        self._state            = _ST_WATCHING
        self._stopped_since    = None
        self._avoid_dir        = 0
        self._avoid_phase_end  = None
        self._avoid_phase      = 0
        self._last_duckie_cx   = None

        # Last detections for debug overlay (list of (x1,y1,x2,y2,score,close))
        self._last_dets        = []

        # ── Model ─────────────────────────────────────────────────────────────
        self.log("Loading ONNX model ...")
        try:
            self.model = MLModel()
            self.log(f"ONNX model loaded. Input: {self.model.net_h}x{self.model.net_w}")
        except FileNotFoundError as e:
            self.logerr(str(e))
            raise

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_info  = rospy.Subscriber(
            "~camera_info", CameraInfo, self._cb_camera_info, queue_size=1)
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self._cb_image,
            queue_size=1, buff_size=2**24)
        self.sub_fsm   = rospy.Subscriber(
            "~mode", FSMState, self._cb_fsm_mode, queue_size=1)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_detected       = rospy.Publisher(
            "~object_detected",        BoolStamped,       queue_size=1)
        self.pub_avoidance_done = rospy.Publisher(
            "~avoidance_done",         BoolStamped,       queue_size=1)
        self.pub_wheels         = rospy.Publisher(
            "~wheels_cmd",             WheelsCmdStamped,  queue_size=1)
        self.pub_debug          = rospy.Publisher(
            "~debug/image/compressed", CompressedImage,   queue_size=1)

        self.log("ObjectDetectionNode ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo):
        if self._img_h is None:
            self._img_h = msg.height
            self._img_w = msg.width
            self.log(f"Camera: {msg.width}x{msg.height}")

    def _cb_fsm_mode(self, msg: FSMState):
        self._fsm_state = msg.state

    def _cb_image(self, msg: CompressedImage):
        if self._fsm_state not in (
                "LANE_FOLLOWING", "OBJECT_DETECTED", "AVOIDING", None):
            return

        try:
            img_bgr = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception as e:
            self.logerr(f"Decode error: {e}")
            return

        orig_h, orig_w = img_bgr.shape[:2]
        if self._img_h is None:
            self._img_h, self._img_w = orig_h, orig_w

        net_h, net_w = self.model.net_h, self.model.net_w
        if orig_h != net_h or orig_w != net_w:
            img_net = cv2.resize(img_bgr, (net_w, net_h))
        else:
            img_net = img_bgr.copy()

        # Run detector
        try:
            dets = self.model._run_detector(img_net)
        except Exception as e:
            self.logerr(f"Inference error: {e}")
            return

        duckie_present, duckie_cx = self._analyse_detections(dets, net_h, net_w)

        if duckie_cx is not None:
            self._last_duckie_cx = duckie_cx

        self._step_state_machine(msg.header, duckie_present)

        # Publish debug image (always, regardless of FSM state)
        if self.pub_debug.get_num_connections() > 0:
            debug_img = self._draw_debug(img_net, dets, net_h, net_w)
            self._publish_debug_image(msg.header, debug_img)

    # ── Detection analysis ─────────────────────────────────────────────────────

    def _analyse_detections(self, dets, net_h, net_w):
        threshold_y = self._stop_thresh.value * net_h
        best_y2 = -1
        best_cx = None
        present = False

        for x1, y1, x2, y2, score, _ in dets:
            if score < self._conf_threshold.value:
                continue
            if y2 > threshold_y:
                present = True
                if y2 > best_y2:
                    best_y2 = y2
                    best_cx = (x1 + x2) / 2.0

        return present, best_cx

    # ── Debug image ────────────────────────────────────────────────────────────

    def _draw_debug(self, img, dets, net_h, net_w):
        """Draw bounding boxes, stop-line and state overlay on a copy of img."""
        out = img.copy()
        threshold_y = int(self._stop_thresh.value * net_h)

        # Draw stop threshold line
        cv2.line(out, (0, threshold_y), (net_w, threshold_y), _COL_LINE, 2)
        cv2.putText(out, f"stop thresh ({self._stop_thresh.value:.0%})",
                    (4, threshold_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COL_LINE, 1)

        # Draw all detections above confidence threshold
        for x1, y1, x2, y2, score, _ in dets:
            if score < self._conf_threshold.value:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            close = (y2 > threshold_y)
            colour = _COL_CLOSE if close else _COL_FAR
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{score:.2f} {'STOP' if close else 'far'}"
            cv2.putText(out, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        # Draw vertical centre line (for avoidance direction reference)
        cx = net_w // 2
        cv2.line(out, (cx, 0), (cx, net_h), (128, 128, 128), 1)

        # State + patience overlay
        if self._state == _ST_STOPPED and self._stopped_since is not None:
            waited = (rospy.Time.now() - self._stopped_since).to_sec()
            patience = self._patience_secs.value
            remaining = max(0.0, patience - waited)
            state_txt = f"STOPPED  avoiding in {remaining:.1f}s"
        elif self._state == _ST_AVOIDING:
            side = "LEFT" if self._avoid_dir > 0 else "RIGHT"
            state_txt = f"AVOIDING ({side})  phase {self._avoid_phase}/3"
        else:
            state_txt = f"WATCHING"

        # Semi-transparent background bar for readability
        bar_h = 22
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (net_w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
        cv2.putText(out, state_txt, (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL_TEXT, 1)

        return out

    def _publish_debug_image(self, header, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data   = buf.tobytes()
        self.pub_debug.publish(msg)

    # ── State machine ──────────────────────────────────────────────────────────

    def _step_state_machine(self, header, duckie_present: bool):
        now = rospy.Time.now()

        if self._state == _ST_WATCHING:
            if duckie_present:
                self.log("Duckie spotted → STOPPED")
                self._state         = _ST_STOPPED
                self._stopped_since = now
                self._publish_detected(header, True)
            else:
                self._publish_detected(header, False)

        elif self._state == _ST_STOPPED:
            if not duckie_present:
                self.log("Duckie gone → WATCHING")
                self._state = _ST_WATCHING
                self._publish_detected(header, False)
                return
            waited = (now - self._stopped_since).to_sec()
            if waited >= self._patience_secs.value:
                self._start_avoidance(header, now)
            else:
                self._publish_detected(header, True)

        elif self._state == _ST_AVOIDING:
            self._run_avoidance(header, now)

    # ── Avoidance ─────────────────────────────────────────────────────────────

    def _start_avoidance(self, header, now):
        net_w = self.model.net_w
        cx    = self._last_duckie_cx
        if cx is None or cx > net_w / 2.0:
            self._avoid_dir = +1
            side = "LEFT"
        else:
            self._avoid_dir = -1
            side = "RIGHT"

        self.log(f"Patience exceeded → avoiding {side} (cx={cx})")
        self._state           = _ST_AVOIDING
        self._avoid_phase     = 1
        self._avoid_phase_end = now + rospy.Duration(self._avoid_steer_secs.value)
        self._publish_detected(header, False)

    def _run_avoidance(self, header, now):
        if now < self._avoid_phase_end:
            self._publish_avoidance_cmd(self._avoid_phase)
            return

        self._avoid_phase += 1
        if self._avoid_phase == 2:
            self.log("Avoidance phase 2: straight")
            self._avoid_phase_end = now + rospy.Duration(self._avoid_fwd_secs.value)
            self._publish_avoidance_cmd(2)
        elif self._avoid_phase == 3:
            self.log("Avoidance phase 3: return")
            self._avoid_phase_end = now + rospy.Duration(self._avoid_return_secs.value)
            self._publish_avoidance_cmd(3)
        else:
            self.log("Avoidance complete → WATCHING")
            self._state          = _ST_WATCHING
            self._last_duckie_cx = None
            self._stopped_since  = None
            done = BoolStamped()
            done.header = header
            done.data   = True
            self.pub_avoidance_done.publish(done)
            self._publish_detected(header, False)

    def _publish_avoidance_cmd(self, phase: int):
        fast = self._avoid_pwm_fast.value
        slow = self._avoid_pwm_slow.value
        fwd  = self._avoid_pwm_fwd.value
        d    = self._avoid_dir
        if phase == 1:
            left  = fast if d < 0 else slow
            right = fast if d > 0 else slow
        elif phase == 2:
            left = right = fwd
        else:
            left  = fast if d > 0 else slow
            right = fast if d < 0 else slow
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.vel_left     = left
        cmd.vel_right    = right
        self.pub_wheels.publish(cmd)

    def _publish_detected(self, header, detected: bool):
        msg = BoolStamped()
        msg.header = header
        msg.data   = bool(detected)
        self.pub_detected.publish(msg)


if __name__ == "__main__":
    node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
