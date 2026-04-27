import numpy as np
import pytest
from src.face_and_color import _clip_rect, extract_cheek_roi,detect_faces_bboxes

def test_clip_rect_basic():
    x, y, w, h = _clip_rect(-10, -10, 50, 50, 100, 100)
    assert x >= 0 and y >= 0 and w > 0 and h > 0

def test_extract_roi_shape():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    face_bbox = (50, 40, 100, 120)
    roi, (x0, y0, W, H) = extract_cheek_roi(img, face_bbox)
    assert roi.shape[0] == H and roi.shape[1] == W
    assert W > 0 and H > 0

def test_invalid_image_raises():
    import src.face_and_color as mod
    with pytest.raises(ValueError):
        mod.detect_faces_bboxes(None)
