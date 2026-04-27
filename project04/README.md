# Personal Color 

## Overview
- 퍼스널 컬러(warm/cool)를 얼굴 중안부 ROI의 LAB 색상 (a, b)로 분류하는 미니 프로젝트입니다.
- 명도 성분 L*은 조명/노출에 민감해 이번 데이터셋에서 성능을 떨어뜨려 제외했고, a*·b*만 사용했습니다.

> 데모/학습 목적으로 조명과 메이크업 영향에 민감하므로 실제 퍼스널 컬러 진단 용도는 권장하지 않습니다.

---

## Data
- 퍼스널 컬러가 알려져 있는 연예진 사진 warm(24) / cool(24) 총 48 장의 데이터
- 데이터에 대한 중간 결과물은 자동으로 `datasets/roi/`, `datasets/processed/feature.csv`에 저장

> Pipeline: 얼굴 탐지 → 중안부 ROI → LAB(a,b) 평균 → SVM
- **얼굴탐지**: MediaPipe(가능 시) → 실패 시 OpenCV Haar fallback
- **ROI**: 입, 치아 영향 최소화를 위해 중안부(midface) 얕고 넓게
- **LAB**: OpenCV LAB(8bit) → a = a_raw-128, b = b_raw-128 (0 중심), L*은 제외

### Process
> 중간 과정에서 데이터 48 -> 45 된 이유, pipline 로그 근거

| 단계 | warm | cool | 합계 | 비고 |
|--|--|--|--|--|
| raw 입력 | 24 | 24 | 48 | `datasets/raws/{warm,cool}` | 
| Face+ROI | 22 | 23 | 45 | no-face ≈ 3 (측면/역광/반사 등) |
| LAB CSV | 22 | 23 | 45 | skipped=0 (읽기/마스크 문제 X) |

- 원인 추정: 
    1. 얼굴 미검풀 (얼굴 각도/빛/반사)
    2. 포맷/회전 이슈
    3. 검풀 임계치

---

## How to Run
```bash
  # 1) ROI 생성
  python -m scripts.extract_rois
  
  # 2) LAB CSV 생성
  python -m scripts.build_features
  
  # 3) SVM
  python -m scripts.train_svm --csv datasets/processed/features.csv --cv 5
  # => models/svm_ab.joblib 저장
  
  # 4) 예측
  python -m scripts.predict_personal --img datasets/raw/warm/sample.jpg --save-vis
  python -m scripts.predict_personal --dir datasets/raw/cool --save-vis
  
  # 5) GUI
  streamlit run app_personal_color.py
```

---

## Results
데이터 45장 (warm 22 / cool 23) 기준 5-fold CV:

| Features | Accuracy | Confusion (rows=true, cols=pred) |
|--|--|--|
| a,b | 0.689 | [[14, 8], [6, 17]]
| L, a,b | 0.644 | [[12, 10], [6, 7]]

### Prediction
> L* 제외하고 a*, b* 로만 예측
- 이번 데이터에서 L 포함시 정확도 0.689 -> 0.644로 하락
- L*은 조명/노출/HDR/하이라이트 영향으로 분산이 커져 클래스 신호를 희석
- 반면 b*(노–파 축)는 웜/쿨 분리에 효과가 컸고(검증 스크립트 기준 Cohen’s d ≈ 1.30)
- a*는 중간 수준 -> ab만 사용이 더 견고

**성공**
| ![alt text](outputs/preds/fall_warm1_warm.jpg) | ![alt text](outputs/preds/spring_warm2_warm.jpg) | ![alt text](outputs/preds/summer_cool1_cool.jpg) | ![alt text](outputs/preds/winter_cool4_cool.jpg) | 
|--|--|--|--|

**실패**
| ![alt text](outputs/preds/fall_warm4_cool.jpg) | ![alt text](outputs/preds/spring_warm12_cool.jpg) | ![alt text](outputs/preds/summer_cool3_warm.jpg) | ![alt text](outputs/preds/winter_cool1_warm.jpg) |
|--|--|--|--|

---

## Project Structure
```text
project04/
├─ datasets/
│  ├─ raws/               # 원본: raws/{warm, cool}/...jpg
│  ├─ roi/                # (생성) 중안부 ROI
│  └─ processed/features.csv
├─ models/svm_ab.joblib   # (생성) SVM 모델(a,b)
├─ src/face_and_color.py  # 얼굴탐지/ROI/LAB 엔진
├─ scripts/
│  ├─ extract_rois.py     # Step1: ROI 생성
│  ├─ build_features.py   # Step2: LAB 특징 CSV
│  ├─ train_svm.py        # Step3: SVM 학습/검증/저장
│  ├─ predict_personal.py # 예측 CLI
│  └─ validate_features.py# CSV 품질 점검
├─ test/
│  ├─ conftest.py         # pytest 실행경로 고정
│  └─ test_face_and_color.py
└─ app_personal_color.py  # Streamlit 간단 GUI
```

---

## Tests
```bash
  pytest -q
```
- `test_face_and_color.py`: `_clip_rect` 안전성, `extract_cheek_roi` 형상 무결성, 잘못된 입력 예외
- `test/conftest.py`: pytest 실행 시 프로젝트 루트를 PYTHONPATH에 주입(모듈 경로 오류 방지)

## Usage
### CLI
```bash
  python -m scripts.predict_personal --img /path/to/me.jpg --save-vis
  python -m scripts.predict_personal --dir datasets/raw/myself --save-vis
  # 결과 CSV: datasets/processed/predictions.csv
  # 시각화: outputs/preds/*_{WARM|COOL}.jpg
```

### GUI
```bash
  streamlit run app_personal_color.py
```
- 좌: 분석 이미지(얼굴/ROI 박스) 
- 우: 웜·쿨 + 정확도 %