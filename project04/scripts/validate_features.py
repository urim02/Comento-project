from pathlib import Path
import argparse, csv
import numpy as np
from collections import defaultdict, Counter

REQUIRED_COLS = ["path", "label", "L", "a", "b", "coverage"]
VALID_LABELS = {"warm", "cool"}

def find_default_csv():
    cands = "./datasets/processed/features.csv"
    return cands[0]

def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"[ERROR] CSV header missing: {missing}")
        for i, row in enumerate(r, 2):
            try:
                L = float(row["L"])
                a = float(row["a"])
                b = float(row["b"])
                cov = float(row["coverage"])
            except Exception:
                print(f"[WARN] numeric parse failed at line {i}: {row}")
                continue
            label = row["label"].strip().lower()
            path = row["path"].strip()
            rows.append({"path": path, "label": label, "L": L, "a": a, "b": b, "coverage": cov})
    return rows

def path_exists(row, base: Path):
    p = Path(row["path"])
    if not p.is_absolute():
        p = base / p
    return p.exists() and p.is_file() and p.stat().st_size > 0

def iqr_bounds(x):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(low), float(high)

def summarize(rows, base_dir: Path, min_cov: float):
    n = len(rows)
    print(f"[INFO] rows: {n}")

    labels = Counter([r["label"] for r in rows])
    unknown = [lab for lab in labels if lab not in VALID_LABELS]
    if unknown:
        print(f"[WARN] unknown labels: {unknown}")
    print(f"[INFO] label counts: {dict(labels)}")

    missing = [r for r in rows if not path_exists(r, base_dir)]
    if missing:
        print(f"[WARN] missing files: {len(missing)} (예: {missing[0]['path']})")
    else:
        print("[OK] all paths exist")

    # 숫자 범위 점검
    Ls = [r["L"] for r in rows]
    As = [r["a"] for r in rows]; Bs = [r["b"] for r in rows]
    C = [r["coverage"] for r in rows]
    def stat(name, arr):
        arr = np.asarray(arr, float)
        print(f"  {name:>9}: mean={arr.mean():.3f} std={arr.std(ddof=1):.3f} min={arr.min():.3f} max={arr.max():.3f}")
    print("[INFO] global stats")
    stat("L", Ls); stat("a", As); stat("b", Bs); stat("coverage", C)

    # 이론 범위 경고
    bad_L = [x for x in Ls if x < 0 or x > 100]
    if bad_L: print(f"[WARN] L out of [0,100]: {len(bad_L)}")
    for name, arr in [("a", As), ("b", Bs)]:
        bad = [x for x in arr if abs(x) > 60]
        if bad: print(f"[WARN] {name} extreme(|x|>60): {len(bad)}")

    # 커버리지 임계치 리포트
    low_cov = [r for r in rows if r["coverage"] < min_cov]
    print(f"[INFO] coverage<{min_cov}: {len(low_cov)} rows")

    # 클래스별 요약
    by_label = defaultdict(list)
    for r in rows: by_label[r["label"]].append(r)
    print("[INFO] per-class stats")
    for lab, lst in by_label.items():
        if lab not in VALID_LABELS: continue
        L = np.array([r["L"] for r in lst]); a = np.array([r["a"] for r in lst]); b = np.array([r["b"] for r in lst])
        print(f"  [{lab}] n={len(lst)}  L_mean={L.mean():.2f}  a_mean={a.mean():.2f}  b_mean={b.mean():.2f}")

    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2: return np.nan
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / s if s > 0 else np.nan
    if all(l in by_label for l in ("warm", "cool")):
        a_d = cohens_d([r["a"] for r in by_label["warm"]], [r["a"] for r in by_label["cool"]])
        b_d = cohens_d([r["b"] for r in by_label["warm"]], [r["b"] for r in by_label["cool"]])
        print(f"[INFO] effect size (Cohen's d): a={a_d:.2f}, b={b_d:.2f}  (0.2≈small, 0.5≈med, 0.8≈large)")

    for name, arr in [("L", Ls), ("a", As), ("b", Bs)]:
        lo, hi = iqr_bounds(arr)
        out = [x for x in arr if x < lo or x > hi]
        print(f"[INFO] outliers by IQR for {name}: {len(out)} (bounds≈[{lo:.2f}, {hi:.2f}])")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=find_default_csv(), help="features.csv 경로")
    ap.add_argument("--min-coverage", type=float, default=0.0, help="리포트 기준 커버리지 임계값(필터링은 안 함)")
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else (Path.cwd() / args.csv)
    base_dir = Path(__file__).resolve().parents[1]  # project04 루트
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        print("[WARN] CSV has 0 valid rows")
        return
    summarize(rows, base_dir, args.min_coverage)

if __name__ == "__main__":
    main()
