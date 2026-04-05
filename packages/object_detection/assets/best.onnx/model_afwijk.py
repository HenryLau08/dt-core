import numpy as np
import onnxruntime as ort
from dt_computer_vision.camera.types import Pixel
from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from solution.config import (
    MODEL_PATH, CONF_THRESHOLD, STOP_DISTANCE,
    FORWARD_PWM, AVOID_PWM,
)

# ── Timing ───────────────────────────────────────────────────────────────────
AVOID_DURATION  = 0.6   # s: hoe lang uitwijken duurt
CENTER_DURATION = 0.5   # s: hoe lang terugsturen duurt
CAMERA_FPS      = 15.0  # frames per seconde van de camera

# ── Rijbaan ───────────────────────────────────────────────────────────────────
# Maximale laterale offset (y) op het grondvlak (in meters).
# Pas aan op jouw rijbaanbreedte.
LANE_Y_MAX      =  0.15  # linker rijbaanrand
LANE_Y_MIN      = -0.15  # rechter rijbaanrand
LANE_MARGIN     =  0.05  # minimale marge van de rand voor "veilig uitwijken"

# ── Stop-detectie ─────────────────────────────────────────────────────────────
# Duckie geldt als "recht voor de auto" als |ground_point.y| < deze drempel.
DIRECT_Y_THRESH = 0.5   # m

# ── Stuurverhouding ───────────────────────────────────────────────────────────
AVOID_TURN_GAIN = 0.25   # 0 = rechtdoor, 1 = draaien op de plaats


class MLModel:
    """
    State machine
    ─────────────
    FORWARD → normaal rijden
    AVOID   → uitwijken om de duckie
    CENTER  → terugsturen naar rijbaanmidden
    STOP    → duckie recht voor ons en te dichtbij, of geen veilige escape
    """

    _ST_FORWARD = "FORWARD"
    _ST_AVOID   = "AVOID"
    _ST_CENTER  = "CENTER"
    _ST_STOP    = "STOP"

    def __init__(self):
        print("Initializing MLModel")
        self.ground_projector = None

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "ONNX model not found (did you download your trained model?):", MODEL_PATH
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype   = np.float16 if inp.type == "tensor(float16)" else np.float32
        self.net_h      = inp.shape[2]
        self.net_w      = inp.shape[3]

        self._state       = self._ST_FORWARD
        self._state_ticks = 0
        self._avoid_dir   = 0   # +1 = rechts, -1 = links

        # Geschatte robot-y op het grondvlak (start in midden rijbaan)
        self._robot_y_est = 0.0

    # ──────────────────────────────────────────────────────────────────────── #
    #  Public API                                                              #
    # ──────────────────────────────────────────────────────────────────────── #

    def set_ground_projector(self, gp):
        self.ground_projector = gp

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return [DifferentialPWM(left=0.0, right=0.0), None]

        pwm = self._step_state_machine(detections)
        self._update_robot_y_estimate(pwm)
        return [pwm, detections]

    # ──────────────────────────────────────────────────────────────────────── #
    #  State machine                                                           #
    # ──────────────────────────────────────────────────────────────────────── #

    def _step_state_machine(self, detections: np.ndarray) -> DifferentialPWM:

        # Getimede staten tellen af ongeacht detecties
        if self._state == self._ST_AVOID:
            self._state_ticks -= 1
            if self._state_ticks <= 0:
                print("State → CENTER")
                self._state       = self._ST_CENTER
                self._state_ticks = self._frames(CENTER_DURATION)
            return self._steer_pwm(self._avoid_dir)

        if self._state == self._ST_CENTER:
            self._state_ticks -= 1
            if self._state_ticks <= 0:
                print("State → FORWARD")
                self._state = self._ST_FORWARD
            return self._steer_pwm(-self._avoid_dir)  # tegenovergestelde kant = terug naar midden

        # FORWARD / STOP: elke frame opnieuw beoordelen
        threat = self._assess_threat(detections)

        if threat is None:
            self._state = self._ST_FORWARD
            return DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM)

        dist, duckie_y = threat

        # Nog ver weg → gewoon doorrijden
        if dist >= STOP_DISTANCE:
            self._state = self._ST_FORWARD
            return DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM)

        # Dichtbij genoeg → beoordeel of de duckie recht voor ons staat
        is_directly_ahead = abs(duckie_y) < DIRECT_Y_THRESH

        if not is_directly_ahead:
            # Duckie staat opzij → we rijden er gewoon langs
            print(f"Duckie opzij (y={duckie_y:.3f} m), doorrijden")
            self._state = self._ST_FORWARD
            return DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM)

        # Duckie staat recht voor ons → probeer uit te wijken
        avoid_dir = self._safe_avoid_direction(duckie_y)

        if avoid_dir == 0:
            # Geen ruimte aan beide kanten → STOP
            print(f"State → STOP (dist={dist:.3f} m, recht voor ons, geen rijbaanruimte)")
            self._state = self._ST_STOP
            return DifferentialPWM(left=0.0, right=0.0)

        print(f"State → AVOID dir={'rechts' if avoid_dir > 0 else 'links'} "
              f"(dist={dist:.3f} m)")
        self._state       = self._ST_AVOID
        self._avoid_dir   = avoid_dir
        self._state_ticks = self._frames(AVOID_DURATION)
        return self._steer_pwm(avoid_dir)

    # ──────────────────────────────────────────────────────────────────────── #
    #  Rijbaanveiligheid                                                       #
    # ──────────────────────────────────────────────────────────────────────── #

    def _safe_avoid_direction(self, duckie_y: float) -> int:
        """
        Kies uitwijkrichting op basis van duckie-positie én beschikbare
        rijbaanruimte (via geschatte robot-y).

        duckie_y > 0 → duckie links  → bij voorkeur rechts uitwijken (robot_y daalt)
        duckie_y < 0 → duckie rechts → bij voorkeur links uitwijken  (robot_y stijgt)

        Geeft +1 (rechts), -1 (links) of 0 (geen veilige kant).
        """
        ruimte_rechts = self._robot_y_est - LANE_Y_MIN   # positief = ruimte aan rechterkant
        ruimte_links  = LANE_Y_MAX - self._robot_y_est   # positief = ruimte aan linkerkant

        heeft_rechts = ruimte_rechts > LANE_MARGIN
        heeft_links  = ruimte_links  > LANE_MARGIN

        if duckie_y >= 0:
            # Duckie links → liefst rechts
            if heeft_rechts:
                return 1
            elif heeft_links:
                return -1   # noodoplossing: toch links
            return 0

        else:
            # Duckie rechts → liefst links
            if heeft_links:
                return -1
            elif heeft_rechts:
                return 1    # noodoplossing: toch rechts
            return 0

    def _update_robot_y_estimate(self, pwm: DifferentialPWM):
        """
        Schat de laterale drift van de robot op basis van het PWM-verschil.
        Vervang door odometrie als dat beschikbaar is.

        Linker wiel sneller dan rechts → auto draait rechts → robot_y daalt.
        """
        dt      = 1.0 / CAMERA_FPS
        dy_gain = 0.04  # m/s per PWM-eenheid verschil (kalibreer op jouw robot)
        delta   = (pwm.left - pwm.right) * dy_gain * dt
        self._robot_y_est = float(np.clip(
            self._robot_y_est + delta,
            LANE_Y_MIN - 0.05,
            LANE_Y_MAX + 0.05,
        ))
        print(f"robot_y_est={self._robot_y_est:.3f} m")

    # ──────────────────────────────────────────────────────────────────────── #
    #  Threat assessment                                                       #
    # ──────────────────────────────────────────────────────────────────────── #

    def _assess_threat(self, detections: np.ndarray):
        """
        Geeft (distance, duckie_ground_y) voor de dichtstbijzijnde detectie,
        of None als er geen bedreiging is.
        """
        best = None

        for x1, y1, x2, y2, score, _ in detections:
            if score < CONF_THRESHOLD:
                continue

            u   = (x1 + x2) / 2
            v   = y2
            pix = Pixel(x=u, y=v)

            try:
                vec          = self.ground_projector.camera.pixel2vector(pix)
                ground_point = self.ground_projector.vector2ground(vec)
                dist         = np.linalg.norm([ground_point.x, ground_point.y])
                print(f"Detection score={score:.2f}  dist={dist:.3f} m  "
                      f"duckie_y={ground_point.y:.3f} m")

                if best is None or dist < best[0]:
                    best = (dist, ground_point.y)

            except Exception as e:
                print(f"Projection error: {e}")

        return best

    # ──────────────────────────────────────────────────────────────────────── #
    #  PWM helpers                                                             #
    # ──────────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _steer_pwm(direction: int) -> DifferentialPWM:
        """
        direction = +1 → rechts sturen (linker wiel sneller)
        direction = -1 → links sturen  (rechter wiel sneller)
        """
        fast = AVOID_PWM
        slow = AVOID_PWM * (1.0 - AVOID_TURN_GAIN)
        if direction >= 0:
            return DifferentialPWM(left=fast, right=slow)
        else:
            return DifferentialPWM(left=slow, right=fast)

    @staticmethod
    def _frames(seconds: float) -> int:
        return max(1, int(seconds * CAMERA_FPS))

    # ──────────────────────────────────────────────────────────────────────── #
    #  ONNX helpers                                                            #
    # ──────────────────────────────────────────────────────────────────────── #

    def _run_detector(self, img_bgr):
        x   = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # [1, N, 6]
        return out[0]

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! "
                f"Expected {self.net_h}x{self.net_w}"
            )
        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img