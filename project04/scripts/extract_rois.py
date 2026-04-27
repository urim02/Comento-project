import cv2
from pathlib import Path
from src.face_and_color import detect_faces_bboxes, extract_cheek_roi

RAW_DIR = Path("./datasets/raw")
OUT_DIR = Path("./datasets/roi")
CLASSES = ["warm", "cool"]
IMG_EXT = {".jpg", ".png"}


def iter_images(d):
    for p in sorted(Path(d).glob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p

def process_split(split):
    print("[DEBUG] scanning:", (RAW_DIR / split).resolve(), "exists=", (RAW_DIR / split).exists())
    in_dir = RAW_DIR / split
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, miss = 0, 0
    for img_path in iter_images(in_dir):
        img = cv2.imread(str(img_path))
        bboxes = detect_faces_bboxes(img)
        if not bboxes:
            miss += 1
            continue
        roi, _ = extract_cheek_roi(img, bboxes[0], mode="midface")
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), roi)
        ok += 1
    return ok, miss

if __name__ == "__main__":
    total_ok, total_miss = 0, 0
    for c in CLASSES:
        ok, miss = process_split(c)
        print(f"[{c}] ROI saved: {ok}, no-face: {miss}")
        total_ok += ok; total_miss += miss
    print(f"==> TOTAL ROI: {total_ok}, misses: {total_miss}")
