# src/config.py
from pathlib import Path

# Roots
HERE = Path(__file__).parent.resolve()   # .../project03/src
ROOT = HERE.parent                       # .../project03

RUN_NAME = "mask_detector_v2"

# Datasets
DATA_YAML    = ROOT / "config" / "data.yaml"
TRAIN_IMAGES = ROOT / "datasets" / "train" / "images"
TRAIN_LABELS = ROOT / "datasets" / "train" / "labels"
VALID_IMAGES = ROOT / "datasets" / "valid" / "images"
VALID_LABELS = ROOT / "datasets" / "valid" / "labels"
TEST_IMAGES  = ROOT / "datasets" / "test"  / "images"
TEST_LABELS  = ROOT / "datasets" / "test"  / "labels"

# Static augmentation outputs
STATIC_AUG_IMAGES = TRAIN_IMAGES.parent / "images_aug"
STATIC_AUG_LABELS = TRAIN_LABELS.parent / "labels_aug"

# Models
WEIGHTS_DIR = ROOT / "weights" / RUN_NAME
MODEL_FILE  = WEIGHTS_DIR / "weights" / "best.pt"
RESULTS_CSV = WEIGHTS_DIR / "results.csv"   # ← 여기만 바꾸면 eval_plot이 항상 최신 결과를 봄
BACKBONE    = HERE / "yolov8s.pt"

# Outputs
GRAPHS      = ROOT / "results" / "graphs"
FINAL_PRED  = ROOT / "results" / "final_predictions"

# Sweep area (파일 폭증은 save=False로 방지)
VAL_SWEEP_PROJ = ROOT / "runs" / "val_sweep"
VAL_SWEEP_NAME = "mask_detector"


# Inference/Eval common
CONF_STAR   = 0.50
IOU_NMS     = 0.45
IMG_SIZE    = 640
RGB_CONVERT = True
SAVE_TXT    = False
