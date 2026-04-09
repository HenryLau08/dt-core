#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
from dt_computer_vision.camera.types import Pixel

from dataclasses import dataclass
from solution.config import MODEL_PATH, CONF_THRESHOLD, STOP_DISTANCE, FORWARD_PWM, AVOID_PWM


@dataclass
class DifferentialPWM:
    """Minimal stand-in for duckietown_messages.actuators.DifferentialPWM.
    Only left/right PWM values are needed by this package."""
    left: float
    right: float

class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.ground_projector = None

        if not MODEL_PATH.exists():
            raise FileNotFoundError("ONNX model not found (did you download your trained model?):", MODEL_PATH)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"], 
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]


    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # shape [1,N,6]
        return out[0]


    def _should_stop(self, detections: np.ndarray): 
        
        stop = False

        for x1, y1, x2, y2, score, _ in detections:

            print(f"Detection: {x1}-{x2}, {y1}-{y2}, {score}")

            # 1. Ignore low-confidence detections
            if score < CONF_THRESHOLD:
                continue

            # 2. Take bottom-center pixel of bounding box
            u = (x1 + x2) / 2
            v = y2

            pix = Pixel(x=u, y=v)

            try:
                # 3. Project pixel to ground plane
                vec = self.ground_projector.camera.pixel2vector(pix)
                ground_point = self.ground_projector.vector2ground(vec)

                # 4. Compute distance (x forward, y lateral)
                distance = np.linalg.norm([ground_point.x, ground_point.y])

                print(f"Distance to duckie: {distance:.3f} m")

                # 5. Check stop condition
                if distance < STOP_DISTANCE:
                    print("STOP: Duckie too close!")
                    stop = True
                    break

            except Exception as e:
                print(f"Projection error: {e}")
                continue

        return stop


    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img


    def set_ground_projector(self, gp):
        self.ground_projector = gp
        

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error {e}")
            return [DifferentialPWM(left=0.0, right=0.0), None]
        if self._should_stop(detections):
            return [DifferentialPWM(left=0.0, right=0.0), detections]
        else:
            return [DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM), detections]
    # def _get_avoidance_pwm(self, detections: np.ndarray):
    #     """
    #     Geeft (left_pwm, right_pwm) terug om om de dichtstbijzijnde duckie heen te sturen.
    #     Geeft None terug als er geen relevante detecties zijn.
    #     """
    #     closest_dist = float('inf')
    #     closest_lateral = None

    #     for x1, y1, x2, y2, score, _ in detections:
    #         if score < CONF_THRESHOLD:
    #             continue

    #         u = (x1 + x2) / 2.0
    #         v = y2
    #         pix = Pixel(x=u, y=v)
    #         vec = self.ground_projector.camera.pixel2vector(pix)
    #         ground_point = self.ground_projector.vector2ground(vec)

    #         distance = np.sqrt(ground_point.x**2 + ground_point.y**2)
    #         if distance < closest_dist:
    #             closest_dist = distance
    #             closest_lateral = ground_point.y  # positief = links, negatief = rechts

    #     if closest_lateral is None:
    #         return None

    #     # Stuur van de duckie weg
    #     AVOID_PWM = 0.0  # hoe sterk je bijstuurt
    #     if closest_lateral > 0:
    #         # Duckie is links → stuur rechts (rechterwielen langzamer)
    #         return (FORWARD_PWM + AVOID_PWM, FORWARD_PWM - AVOID_PWM)
    #     else:
    #         # Duckie is rechts → stuur links (linkerwielen langzamer)
    #         return (FORWARD_PWM - AVOID_PWM, FORWARD_PWM + AVOID_PWM)

    # def get_wheel_velocities_from_image(self, img: np.ndarray):
    #     try:
    #         detections = self._run_detector(img)
    #     except Exception as e:
    #         print(f"ONNX inference error {e}")
    #         return [DifferentialPWM(left=0.0, right=0.0), None]

    #     if self._should_stop(detections):
    #         # Probeer te ontwijken in plaats van volledig te stoppen
    #         avoidance = self._get_avoidance_pwm(detections)
    #         if avoidance:
    #             left, right = avoidance
    #             return [DifferentialPWM(left=left, right=right), detections]
    #         else:
    #             return [DifferentialPWM(left=0.0, right=0.0), detections]
    #     else:
    #         return [DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM), detections]
