import cv2
import numpy as np

# 이미지 로드 + 예외 처리
img = cv2.imread('image.jpeg')
if img is None:
    print("[Warning] Failed to load image: image.jpeg")
    exit(1)

# BGR -> HSV 변환 + 예외 처리
try:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
except cv2.error as e:
    print(f"[Error] 컬러 변환 실패: {e}")
    exit(1)

# 트랙바
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
for name, init, maxv in [
    ('H1 Low',   0, 179),
    ('H1 High', 10, 179),
    ('H2 Low', 170, 179),
    ('H2 High',179, 179),
    ('Sat Min',120, 255),
    ('Val Min', 70, 255),
]:
    cv2.createTrackbar(name, 'Settings', init, maxv, lambda x: None)

while True:
    h1l = cv2.getTrackbarPos('H1 Low',   'Settings')
    h1h = cv2.getTrackbarPos('H1 High',  'Settings')
    h2l = cv2.getTrackbarPos('H2 Low',   'Settings')
    h2h = cv2.getTrackbarPos('H2 High',  'Settings')
    s_l = cv2.getTrackbarPos('Sat Min',  'Settings')
    v_l = cv2.getTrackbarPos('Val Min',  'Settings')

    lower1 = np.array([h1l, s_l, v_l])
    upper1 = np.array([h1h, 255, 255])
    lower2 = np.array([h2l, s_l, v_l])
    upper2 = np.array([h2h, 255, 255])

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask  = mask1 + mask2

    result = cv2.bitwise_and(img, img, mask=mask)

    combined = np.hstack((img, result))
    cv2.imshow('Original | Red Filtered', combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
