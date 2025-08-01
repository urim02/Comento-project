import cv2

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러오지 못했습니다: {path}")

    return image

def cvt_to_hsv(image):
    if image is None:
        raise ValueError("입력 이미지가 None입니다.")
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        raise ValueError(f"HSV 변환 실패: {e}")

