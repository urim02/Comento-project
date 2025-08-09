# src/train.py
import cv2, shutil, torch
import albumentations as A
from ultralytics import YOLO
from config import (
    DATA_YAML, TRAIN_IMAGES, TRAIN_LABELS,
    STATIC_AUG_IMAGES, STATIC_AUG_LABELS,
    BACKBONE, RUN_NAME, ROOT
)

# ---- 하이퍼 ----
EPOCHS  = 20
BATCH   = 8
LR0     = 0.001
OPTIM   = "AdamW"
IOU_VAL = 0.45
DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"

def make_static_augment():
    STATIC_AUG_IMAGES.mkdir(parents=True, exist_ok=True)
    STATIC_AUG_LABELS.mkdir(parents=True, exist_ok=True)

    # Affine로 경고 제거 + 과도한 왜곡 방지
    transform = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.2
    ))

    MIN_AREA = 0.0016  # 너무 작은 박스 제거(0.16%)

    def clip_xywh(box):
        x, y, w, h = box
        x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0); h = min(max(h, 1e-6), 1.0)
        return [x, y, w, h]

    for img_path in list(TRAIN_IMAGES.glob("*.jpg")) + list(TRAIN_IMAGES.glob("*.png")):
        base = img_path.stem
        lbl_path = TRAIN_LABELS / f"{base}.txt"
        if not lbl_path.exists():
            continue

        # 원본 포함
        shutil.copy(img_path, STATIC_AUG_IMAGES / f"{base}.jpg")
        shutil.copy(lbl_path, STATIC_AUG_LABELS / f"{base}.txt")

        lines = [l.split() for l in lbl_path.read_text().splitlines() if l.strip()]
        classes, bboxes = [], []
        for l in lines:
            if len(l) != 5:
                continue
            cls = int(float(l[0]))
            x, y, w, h = map(float, l[1:])
            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                classes.append(cls)
                bboxes.append([x, y, w, h])

        if not bboxes:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i in range(2):  # AUG_PER_IMAGE=2
            out = transform(image=img, bboxes=bboxes, class_labels=classes)
            clean_boxes, clean_cls = [], []
            for cls, box in zip(out["class_labels"], out["bboxes"]):
                box = clip_xywh(box)
                if box[2] * box[3] >= MIN_AREA:
                    clean_boxes.append(box); clean_cls.append(int(cls))
            if not clean_boxes: continue
            out_base = f"{base}_aug{i}"
            cv2.imwrite(str(STATIC_AUG_IMAGES / f"{out_base}.jpg"), out["image"])
            with open(STATIC_AUG_LABELS / f"{out_base}.txt", "w") as fw:
                for cls, box in zip(clean_cls, clean_boxes):
                    fw.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")
    print("✅ Static augmentation complete (valid boxes only).")

if __name__ == "__main__":
    make_static_augment()
    print("data.yaml 의 train 경로에 images_aug 포함되어 있는지 확인하세요.")

    model = YOLO(str(BACKBONE))  # COCO 사전학습
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        optimizer=OPTIM,
        lr0=LR0, lrf=0.01,
        project=str(ROOT / "weights"),
        name=RUN_NAME,           # == mask_detector_v2
        exist_ok=True,
        device=DEVICE,
        workers=4,
        cache="disk",
        iou=IOU_VAL,
        conf=0.25,
        max_det=200,
        resume=False,
        patience=15,
        verbose=True
    )
    print("device:", DEVICE)
