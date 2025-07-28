# Red Detection 

-----

## Overview
> OpenCV를 활용해 단일 이미지에서 **빨간색 픽셀만** 추출하는 과제입니다. <br>

- **목표**: HSV 색공간에서 빨간색 범위를 지정하여 마스크 생성 <br> -> 원본에서 빨간색 영역만 필터링


- **예외 처리**:
  - 'cv.imread' 실패 시 경고 후 종료
  - 'cv2.cvtColor' 실패 시 경고 후 종료

## Pipline Details
1. **Image Load & Validation**

```python
img = cv2.imread('image.jpeg')
if img is None:
    print("[Warning] Failed to load image: image.jpeg")
    exit(1)
````

2. **BGR -> HSV**

```python
try:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
except cv2.error as e:
    print(f"[Error] 컬러 변환 실패: {e}")
    exit(1)
```

3. **Red Mask 생성**
- 빨간색 특성상 두 개의 Hue 범위 생성

```python
mask1 = cv2.inRange(img_hsv, lower1, upper1)
mask2 = cv2.inRange(img_hsv, lower2, upper2)
mask  = mask1 + mask2
```

## Results
> Original | Red Filtered
- 원본에서 빨간색 영역 필터링

## How to Run
1. 패키지 설치
```bash
  pip install opencv-python numpy
```

2. 스크립트 실행
```bash
  python image_filtering.py
```

3. 종료
- 창 표시 모드에서 ESC 키를 누르면 종료됩니다.
