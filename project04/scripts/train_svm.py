from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
import joblib

def load_df(csv_path: Path):
    df = pd.read_csv(csv_path)
    need = {"label","a","b"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[ERROR] CSV missing columns: {need - set(df.columns)}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("./datasets/processed/features.csv"))
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--tmin", type=float, default=0.35)
    ap.add_argument("--tmax", type=float, default=0.65)
    ap.add_argument("--tstep", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path("./models/svm_ab.joblib"))
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    csv_path = args.csv if args.csv.is_absolute() else (base / args.csv)
    df = load_df(csv_path)

    X = df[["a","b"]].values.astype(np.float32)
    y = df["label"].values
    print(f"[INFO] using features: ['a','b'], n={len(df)}")

    # class_weight 제거 + 균형정확도로 최적화
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])
    param_grid = {
        "svc__C":     [0.1, 0.3, 1, 3, 10, 30, 100],
        "svc__gamma": ["scale", "auto", 0.3, 0.1, 0.03, 0.01, 0.003]
    }
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"[INFO] best params: {gs.best_params_}")
    print(f"[INFO] best CV balanced_acc: {gs.best_score_:.3f}")

    # 최적 파이프라인으로 CV 예측 -> 임계값 튜닝
    best_pipe = gs.best_estimator_
    proba = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    best_pipe.fit(X, y)
    classes = list(best_pipe.named_steps["svc"].classes_)
    warm_idx = classes.index("warm")
    p_warm = proba[:, warm_idx]

    # 임계값 sweep
    ts = np.arange(args.tmin, args.tmax + 1e-9, args.tstep)
    best_t, best_f1, best_bacc = 0.5, -1.0, -1.0
    for t in ts:
        y_hat = np.where(p_warm >= t, "warm", "cool")
        f1 = f1_score(y, y_hat, average="macro")
        bacc = balanced_accuracy_score(y, y_hat)
        if f1 > best_f1:
            best_f1, best_bacc, best_t = f1, bacc, t

    print(f"[TUNE] best threshold t={best_t:.3f} → macroF1={best_f1:.3f}, balanced_acc={best_bacc:.3f}")

    # 임계값으로 최종 CV 리포트
    y_hat = np.where(p_warm >= best_t, "warm", "cool")
    acc = accuracy_score(y, y_hat)
    bacc = balanced_accuracy_score(y, y_hat)
    print(f"[CV] accuracy={acc:.3f}, balanced_acc={bacc:.3f}")
    print("[CV] confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, y_hat, labels=["warm","cool"]))
    print("[CV] classification report:")
    print(classification_report(y, y_hat, target_names=["warm","cool"]))

    # 전체 데이터로 재학습 + 임계값 저장
    best_pipe.fit(X, y)
    out_path = args.out if args.out.is_absolute() else (base / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": best_pipe, "features": ["a","b"], "labels": classes, "threshold": float(best_t)}
    joblib.dump(payload, out_path)
    print(f"[SAVE] {out_path.resolve()}")

if __name__ == "__main__":
    main()
