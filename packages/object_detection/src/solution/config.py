from pathlib import Path

# parents[2] = dt-core/packages/object_detection/
# assets/best.onnx lives inside that package folder
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR   = PACKAGE_ROOT / "assets"

MODEL_PATH   = ASSETS_DIR / "best.onnx"
CLASSES_YAML = ASSETS_DIR / "classes.yaml"

# ── Detection thresholds ──────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5   # minimum confidence score to act on a detection
STOP_DISTANCE  = 0.5   # metres: stop when duckie is closer than this
AVOID_DISTANCE = 0.8   # metres: start avoidance manoeuvre when closer than this

# ── Wheel PWM values ──────────────────────────────────────────────────────────
FORWARD_PWM = 0.4   # straight-line driving
AVOID_PWM   = 0.15  # used during avoidance steering

# ── Data-collection helpers (not used by the ROS node) ───────────────────────
DATA_DIR             = Path("/data/")
DATA_COLLECTION_ROOT = DATA_DIR / "data_collection"
DATASET_DIR          = DATA_DIR / "duckietown_dataset"
TRAIN_DIR            = DATASET_DIR / "train"
VAL_DIR              = DATASET_DIR / "val"

SAVE_EVERY_N_FRAMES = 3
MAX_LOG_IMAGES      = 1000
