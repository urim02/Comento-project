import argparse, glob, sys
from pathlib import Path
import cv2
from ultralytics import YOLO

from config import (
    MODEL_FILE, TEST_IMAGES, FINAL_PRED,
    CONF_STAR, IOU_NMS, IMG_SIZE, RGB_CONVERT
)

# 클래스별 임계치(오탐 줄이고, 마스크 계열 놓침 회수)
CLASS_THRESH = {
    "mask_weared":    0.50,
    "mask_incorrect": 0.55,
    "no_mask":        0.60,  # 오탐 억제
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conf_raw", type=float, default=0.25, help="모델에서 1차로 뽑아올 최소 conf(후처리로 다시 거름)")
    p.add_argument("--conf_star", type=float, default=CONF_STAR, help="기본 운영 임계치(클래스별 임계치 없는 경우에 사용)")
    p.add_argument("--iou", type=float, default=IOU_NMS)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--fallback", action="store_true", help="1차에서 박스 0개면 2차 낮은 conf로 재시도")
    p.add_argument("--fallback_conf", type=float, default=0.40)
    p.add_argument("--limit", type=int, default=0, help="테스트 이미지 N장만 시도(0=전체)")
    return p.parse_args()

def keep_by_class(results, default_conf):
    """클래스별 임계치로 필터링"""
    kept = []
    names = results.names
    for b in results.boxes:
        cls_id = int(b.cls[0])
        name   = names[cls_id]
        conf   = float(b.conf[0])
        th     = CLASS_THRESH.get(name, default_conf)
        if conf >= th:
            kept.append(b)
    return kept

def draw_and_save(orig_bgr, boxes, names, out_path):
    canvas = orig_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls[0]); label = names[cls_id]
        score  = float(b.conf[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(canvas, f"{label} {score:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    ok = cv2.imwrite(str(out_path), canvas)
    return ok

def main():
    args = parse_args()

    # 모델 로드
    if not MODEL_FILE.exists():
        print(f"[ERR] 모델 없음: {MODEL_FILE}"); sys.exit(1)
    model = YOLO(str(MODEL_FILE))
    print(f"[OK] Loaded: {MODEL_FILE}")

    # 출력 폴더
    FINAL_PRED.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {FINAL_PRED}")

    # 테스트 이미지 로드
    img_paths = sorted(glob.glob(str(TEST_IMAGES / "*.jpg")))
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    if not img_paths:
        print(f"[ERR] 테스트 이미지 없음: {TEST_IMAGES}"); sys.exit(1)
    print(f"[INFO] {len(img_paths)} images")

    for p in img_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[WARN] 로딩 실패: {p}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if RGB_CONVERT else bgr

        # 1차: 낮은 conf_raw로 넉넉히 뽑아오고 → 클래스별 임계치로 거름
        res1 = model.predict(rgb, conf=args.conf_raw, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        kept = keep_by_class(res1, default_conf=args.conf_star)

        # 2차(선택): 아무 것도 못 찾았을 때만 낮춘 conf로 재시도
        if args.fallback and len(kept) == 0:
            res2 = model.predict(rgb, conf=args.fallback_conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
            kept = keep_by_class(res2, default_conf=args.conf_star)

        out_path = FINAL_PRED / Path(p).name
        ok = draw_and_save(bgr, kept, res1.names, out_path)
        print(f"[SAVE] {out_path.name} ({len(kept)} boxes)" if ok else f"[FAIL] {out_path.name}")

    print("[DONE] inference finished.")

if __name__ == "__main__":
    main()
