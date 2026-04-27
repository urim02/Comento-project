from pathlib import Path
import argparse, csv
import cv2, joblib
import numpy as np
from src.face_and_color import detect_faces_bboxes, extract_cheek_roi, roi_to_lab_feature

def load_payload(model_path: Path):
    payload = joblib.load(model_path)
    pipe = payload["model"]
    feats = payload.get("features", ["a","b"])
    labels = payload.get("labels", list(pipe.named_steps["svc"].classes_))
    thr = float(payload.get("threshold", 0.5))
    warm_idx = labels.index("warm") if "warm" in labels else 0
    return pipe, feats, labels, thr, warm_idx

def predict_one(img_path: Path, pipe, feats, thr, warm_idx, mode="midface", save_vis=False, vis_dir: Path=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"path": str(img_path), "error": "read_fail"}

    bboxes = detect_faces_bboxes(img)
    if not bboxes:
        return {"path": str(img_path), "error": "no_face"}

    face = bboxes[0]
    roi, (x0, y0, W, H) = extract_cheek_roi(img, face, mode=mode)
    Lm, am, bm, cov = roi_to_lab_feature(roi, use_skin_mask=True)

    if feats == ["a","b"]:
        X = np.array([[am, bm]], dtype=np.float32)
    elif feats == ["L","a","b"]:
        X = np.array([[Lm, am, bm]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported features: {feats}")

    proba = pipe.predict_proba(X)[0]
    p_warm = float(proba[warm_idx])
    pred = "warm" if p_warm >= thr else "cool"
    conf = p_warm if pred == "warm" else (1.0 - p_warm)

    out = {
        "path": str(img_path),
        "pred": pred,
        "conf": round(conf, 4),
        "p_warm": round(p_warm, 4),
        "L": round(float(Lm), 4),
        "a": round(float(am), 4),
        "b": round(float(bm), 4),
        "coverage": round(float(cov), 4),
        "error": ""
    }

    if save_vis:
        vis = img.copy()
        # 얼굴 탐지 박스
        fx, fy, fw, fh = face
        cv2.rectangle(vis, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
        # ROI bbox
        cv2.rectangle(vis, (x0, y0), (x0+W, y0+H), (255, 0, 0), 2)
        txt = f"{pred.upper()} ({conf:.2f})"
        cv2.putText(vis, txt, (fx, max(0, fy-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,255), 2, cv2.LINE_AA)
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / (img_path.stem + f"_{pred}.jpg")
        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            return {"path": str(img_path), "error": "imwrite_fail"}

    return out

def iter_images(root: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    if root.is_file():
        yield root
    else:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in exts and p.is_file():
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, help="단일 이미지 경로")
    ap.add_argument("--dir", type=Path, help="폴더 경로")
    ap.add_argument("--model", type=Path, default=Path("./models/svm_ab.joblib"))
    ap.add_argument("--mode", type=str, default="midface", choices=["midface","center","dual"])
    ap.add_argument("--save-vis", action="store_true", help="시각화 이미지 저장")
    ap.add_argument("--vis-dir", type=Path, default=Path("./outputs/preds"))
    ap.add_argument("--out-csv", type=Path, default=Path("./datasets/processed/predictions.csv"))
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    model_path = args.model if args.model.is_absolute() else (base/args.model)
    pipe, feats, labels, thr, warm_idx = load_payload(model_path)
    print(f"[INFO] model={model_path.name}, features={feats}, threshold={thr:.3f}")

    targets = []
    if args.img: targets.append(args.img if args.img.is_absolute() else (base/args.img))
    if args.dir: targets.extend(iter_images(args.dir if args.dir.is_absolute() else (base/args.dir)))
    if not targets:
        raise SystemExit("Specify --img or --dir")

    results = []
    for p in targets:
        out = predict_one(p, pipe, feats, thr, warm_idx, mode=args.mode,
                          save_vis=args.save_vis, vis_dir=(base/args.vis_dir))
        results.append(out)
        if out["error"]:
            print(f"[ERR] {p}: {out['error']}")
        else:
            print(f"[OK] {p.name}: {out['pred']} ({out['conf']:.2f})  a={out['a']} b={out['b']} cov={out['coverage']}")

    # 저장
    out_csv = args.out_csv if args.out_csv.is_absolute() else (base/args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","pred","conf","p_warm","L","a","b","coverage","error"])
        w.writeheader()
        w.writerows(results)
    print(f"[SAVE] {out_csv.resolve()}  (n={len(results)})")

if __name__ == "__main__":
    main()
