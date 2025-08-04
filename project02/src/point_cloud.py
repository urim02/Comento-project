import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import load_image
from depth_map import generate_depth_map


def generate_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape[:2]

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # grayscale로 깊이값 변환
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    z_coords = gray.astype(np.float32) * scale

    # (x, y, z) 포인트 생성
    points_3d = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)

    return points_3d


def plot_point_cloud(points):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, s=0.5, c=zs, cmap='jet')
    ax.set_title("3D Point Cloud from Depth Map")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth (Z)')
    plt.show()


if __name__ == "__main__":
    image_path = "../original.jpeg"

    try:
        image = load_image(image_path)
        depth_map = generate_depth_map(image)
        points = generate_point_cloud(depth_map)
        plot_point_cloud(points)
    except Exception as e:
        print(f"[ERROR] 처리 실패: {e}")
