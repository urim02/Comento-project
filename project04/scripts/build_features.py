from pathlib import Path
import csv
import cv2
from src.face_and_color import roi_to_lab_feature

ROI_DIR = Path("./datasets/roi")
OUT_CSV = Path("./datasets/processed/features.csv")
CLASSES = ["warm", "cool"]
IMG_EXT = {".jpg", ".png"}

MIN_COVERAGE = 0.0

print("[DEBUG] ROI_DIR ->", ROI_DIR.resolve(), "exists=", ROI_DIR.exists())
for cls in ["warm", "cool"]:
    p = ROI_DIR / cls
    print("[DEBUG]", cls, "dir ->", p.resolve(), "exists=", p.exists())
    if p.exists():
        from itertools import islice
        sample = list(islice((x for x in p.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}), 3))
        print("[DEBUG] sample files:", [str(s.name) for s in sample])

def iter_images(d: Path):
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p

def main():
    rows = []
    total, dropped = 0, 0

    for label in CLASSES:
        in_dir = ROI_DIR / label
        if not in_dir.exists():
            print(f"[WARN] missing ROI dir: {in_dir.resolve()}")
            continue

        ok, skip = 0, 0
        for p in iter_images(in_dir):
            total += 1
            img = cv2.imread(str(p))
            if img is None:
                skip += 1
                continue
            try:
                Lm, am, bm, cov = roi_to_lab_feature(img, use_skin_mask=True)
                if cov < MIN_COVERAGE:
                    dropped += 1
                    continue
                rows.append({
                    "path": str(p),
                    "label": label,
                    "L": round(Lm, 4),
                    "a": round(am, 4),
                    "b": round(bm, 4),
                    "coverage": round(cov, 4)
                })
                ok += 1
            except Exception:
                skip += 1
        print(f"[{label}] features: {ok}, skipped: {skip}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "L", "a", "b", "coverage"])
        w.writeheader()
        w.writerows(rows)

    print(f"==> saved: {OUT_CSV.resolve()} ({len(rows)} rows, total_in={total}, dropped={dropped})")

if __name__ == "__main__":
    main()