import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import DATA_YAML, MODEL_FILE, RESULTS_CSV, GRAPHS, VAL_SWEEP_PROJ, VAL_SWEEP_NAME, CONF_STAR, IMG_SIZE, IOU_NMS

GRAPHS.mkdir(parents=True, exist_ok=True)
(VAL_SWEEP_PROJ / VAL_SWEEP_NAME).mkdir(parents=True, exist_ok=True)

# 파일 체크
for p in (DATA_YAML, MODEL_FILE, RESULTS_CSV):
    if not p.exists():
        print(f"파일이 없습니다: {p}")
        sys.exit(1)

# 모델 평가
model   = YOLO(str(MODEL_FILE))
metrics = model.val(
     data=str(DATA_YAML),
     conf=CONF_STAR,     # 운영 임계치
     imgsz=IMG_SIZE,     # 학습과 동일
     iou=IOU_NMS,        # NMS 기준 통일
     save=False, plots=False
)
prec0, rec0, m50, _ = metrics.mean_results()
print(f"> Baseline (conf={CONF_STAR:.2f}) → P:{prec0:.3f} R:{rec0:.3f} mAP50:{m50:.3f}")
# 학습 로그(csv) -> epoch별 PR
df     = pd.read_csv(RESULTS_CSV)
epochs = df["epoch"] if "epoch" in df.columns else (df.index + 1)

plt.figure()
plt.plot(epochs, df["metrics/precision(B)"], label="Precision")
plt.plot(epochs, df["metrics/recall(B)"],    label="Recall")
plt.xlabel("Epoch"); plt.ylabel("Score")
plt.title("Precision & Recall Over Epochs")
plt.legend(); plt.grid(True)
out1 = GRAPHS / "performance_curve.png"
plt.savefig(out1)
print(f"> Epoch별 그래프 저장: {out1}")
plt.close()

# Threshold sweep -> PR curve
thresholds = [i/20 for i in range(1,20)]
confs_used  = []
precisions  = []
recalls     = []

print("> Threshold sweep 시작")
for th in thresholds:              # th는 후보 임계치
    m = model.val(
        data=str(DATA_YAML),
        conf=th, imgsz=IMG_SIZE, iou=IOU_NMS,
        save=False, plots=False,
        project=str(VAL_SWEEP_PROJ), name=VAL_SWEEP_NAME, exist_ok=True
    )
    p, r, _, _ = m.mean_results()
    confs_used.append(th)          # ← 실제로 쓴 값 기록
    precisions.append(p)
    recalls.append(r)
    print(f"conf={th:.2f} → P={p:.3f}, R={r:.3f}")

confs_used = np.array(confs_used)
precs = np.array(precisions)
recs  = np.array(recalls)
f1s   = 2 * (precs * recs) / (precs + recs + 1e-9)
best  = int(f1s.argmax())
print(f"[F1-최대] conf={confs_used[best]:.2f}  P={precs[best]:.3f}  R={recs[best]:.3f}  F1={f1s[best]:.3f}")

plt.figure(figsize=(6,6))
plt.plot(recalls, precisions, marker='o')
for x, y, t in zip(recalls, precisions, thresholds):
    plt.text(x, y, f"{t:.2f}", fontsize=8, ha='right', va='bottom')
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Threshold Sweep)")
plt.grid(True)
out2 = GRAPHS / "pr_curve_threshold_sweep.png"
plt.savefig(out2)
print(f"> PR 곡선 저장: {out2}")
plt.show()
