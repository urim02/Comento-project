import cv2
import os
import numpy as np

def generate_depth_map(img):
    if img is None:
        raise ValueError("입력 이미지가 None입니다.")

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return depth_map
    except cv2.error as e:
        raise ValueError(f"Depth Map 변환 실패: {e}")

if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)

    img_path = "../original.jpeg"
    img = cv2.imread(img_path)
    depth_map = generate_depth_map(img)

    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))

    combined = np.hstack((img, depth_map))

    cv2.imshow("Original | Depth Map", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 파일로 저장
    combined = np.hstack((img, depth_map))
    cv2.imwrite("../result/combined_output.jpg", combined)

