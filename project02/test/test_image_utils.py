import numpy as np
import pytest
import cv2
from project02.src.image_utils import load_image, cvt_to_hsv

def test_load_image_success(tmp_path):
    path = tmp_path / "test.jpg"
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(path), test_img)

    image = load_image(str(path))
    assert image.shape == (100, 100, 3)

def test_load_image_fail():
    with pytest.raises(FileNotFoundError):
        load_image("없는파일.jpg")

def test_cvt_to_hsv_valid():
    test_img = np.zeros((50, 50, 3), dtype=np.uint8)
    hsv = cvt_to_hsv(test_img)
    assert hsv.shape == test_img.shape
    assert hsv.dtype == test_img.dtype

def test_cvt_to_hsv_invalid():
    with pytest.raises(ValueError):
        cvt_to_hsv(None)
