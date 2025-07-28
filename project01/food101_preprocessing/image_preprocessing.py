from datasets import load_dataset
from PIL import Image
import numpy as np
import cv2
import os
import torchvision.transforms as T

# parameter
BRIGHTNESS_THRESHOLD = 0.1
AREA_RATIO_THRESHOLD = 0.05
OUTPUT_DIR           = "preprocessed_samples"
NUM_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# augmentation
aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2,
                  contrast=0.2,
                  saturation=0.2)
])

def img_preprocessing(img: Image.Image):
    # resize -> grayscale -> normalize
    gray = img.resize((224,224)).convert("L")
    arr  = np.array(gray, dtype=np.float32) / 255.0

    # 너무 어두운 이미지 제거 / 객체 너무 작으면 제거
    if arr.mean() < BRIGHTNESS_THRESHOLD:
        return None

    _, mask = cv2.threshold(
        (arr * 255).astype(np.uint8),
        int(BRIGHTNESS_THRESHOLD * 255),
        255,
        cv2.THRESH_BINARY
    )
    if (mask > 0).sum() / mask.size < AREA_RATIO_THRESHOLD:
        return None

    # Gaussian blur
    blurred = cv2.GaussianBlur(mask, (5,5), 0)
    return blurred

ds = load_dataset("ethz/food101", split=f"train[:NUM_SAMPLES]")

for idx, sample in enumerate(ds):
    orig_pil = sample["image"]
    blurred = img_preprocessing(orig_pil)
    if blurred is None:
        print(f"{idx:02d}: filtered out (너무 어둡거나 작음)")
        continue

    # 증강 - 컬러 원본
    orig_resized = orig_pil.resize((224,224))
    aug_pil      = aug(orig_resized)

    # Numpy 변환
    orig_np = cv2.cvtColor(np.array(orig_resized), cv2.COLOR_RGB2BGR)
    blur_np = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    aug_np  = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)

    combined = np.hstack((orig_np, blur_np, aug_np))

    out_path = os.path.join(OUTPUT_DIR, f"combined_{idx:02d}.png")
    cv2.imwrite(out_path, combined)
    print(f"{idx:02d}: saved → {out_path}")
