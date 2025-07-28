import cv2
import numpy as np

img = cv2.imread('./image.jpeg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)

cv2.createTrackbar('H1 Low', 'Settings', 0, 179, lambda x: None)
cv2.createTrackbar('H1 High', 'Settings', 10, 179, lambda x: None)
cv2.createTrackbar('H2 Low', 'Settings', 170, 179, lambda x: None)
cv2.createTrackbar('H2 High', 'Settings', 179, 179, lambda x: None)
cv2.createTrackbar('Sat Min', 'Settings', 120, 255, lambda x: None)
cv2.createTrackbar('Val Min', 'Settings', 70, 255, lambda x: None)

while True:
    h1_low  = cv2.getTrackbarPos('H1 Low', 'Settings')
    h1_high = cv2.getTrackbarPos('H1 High', 'Settings')
    h2_low  = cv2.getTrackbarPos('H2 Low', 'Settings')
    h2_high = cv2.getTrackbarPos('H2 High', 'Settings')
    sat_min = cv2.getTrackbarPos('Sat Min', 'Settings')
    val_min = cv2.getTrackbarPos('Val Min', 'Settings')

    lower1 = np.array([h1_low, sat_min, val_min])
    upper1 = np.array([h1_high, 255, 255])
    lower2 = np.array([h2_low, sat_min, val_min])
    upper2 = np.array([h2_high, 255, 255])

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask = mask1 + mask2

    result = cv2.bitwise_and(img, img, mask=mask)

    combined = np.hstack((img, result))
    cv2.imshow('Original | Red Filtered', combined)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

