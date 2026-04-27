# Comento Image Processing

> 단일 이미지 빨간색 검출 과제와
> Hugging Face Food101 데이터셋 전처리 / 필터링 과제를 한 곳에 정리합니다.

-----

## Overview
1. **Red Detection (1차 과제)**
    - OpenCV를 활용해 단일 이미지에서 빨간색 픽셀만 추출
    - HSV 색공간 변환, 두 개의 Hue 범위 마스크
    - 실시간 GUI 창 + 예외 처리(이미지 로드 / 색 변환)

2. **Food101 Preprocessing**
    - Hugging Face 'ethz/food101' 데이터셋 전처리
      - 기본: Resize -> Grayscale & Normalize -> Gaussian Blur -> Augment
      - 심화: 어두운 이미지 / 작은 객체 필터링

## Prerequisites
- (권장) 가상환경
- 필수 패키지:
  ```bash
  pip install opencv-python numpy datasets pillow torchvision
  ```
  
## Project Structure
```text
Comento-project/
├── README.md
└── project01/
    ├── red_detection/
    │   ├── image_filtering.py
    │   ├── sample.jpg
    │   └── README.md            ← Red Detection 설명
    └── food101_preprocessing/
        ├── image_preprocessing.py
        ├── image_filtering.py
        ├── preprocessed_samples/
        └── README.md            ← Food101 Preprocessing 설명
```

## How to Run
1. **Red Detection**
    ```bash
   cd project01/red_detection
   python image_filtering.py
    ```
   
2. **Food101 Preprocessing**
    ```bash
    cd project01/food101_preprocessing
    python image_preprocessing.py
    ```