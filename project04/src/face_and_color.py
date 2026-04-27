import cv2
import numpy as np

# Mediapipe 우선 사용
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    _mp_fd = mp.solutions.face_detection
except Exception:
    _MP_AVAILABLE = False

def _clip_rect(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def detect_faces_bboxes(img_bgr):
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("입력 이미지가 올바르지 않음")

    H, W = img_bgr.shape[:2]
    boxes = []

    if _MP_AVAILABLE:
        with _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = fd.process(rgb)
            if res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * W)
                    y = int(bb.ymin * H)
                    w = int(bb.width * W)
                    h = int(bb.height * H)
                    x, y, w, h = _clip_rect(x, y, w, h, W, H)
                    boxes.append((x, y, w, h))
    else:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in detections:
            x, y, w, h = _clip_rect(x, y, w, h, W, H)
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes

def extract_cheek_roi(img_bgr, face_bbox, mode="center"):
    H_img, W_img = img_bgr.shape[:2]
    x, y, w, h = face_bbox

    cx, cy = x + w / 2, y + h / 2
    scale = 1.10
    w2, h2 = int(w * scale), int(h * scale)
    x = int(cx - w2 / 2);
    y = int(cy - h2 / 2);
    w = w2;
    h = h2
    x, y, w, h = _clip_rect(x, y, w, h, W_img, H_img)

    if mode == "midface":
        # 중안부 비율 - 입, 치아 제외
        top_r, bot_r = 0.38, 0.68                           # 세로 범위
        width_r = 0.80                                      # 가로 폭
        W = int(w * width_r)
        H = int(h * (bot_r - top_r))
        x0 = x + (w - W) // 2
        y0 = y + int(h * top_r)
        x0, y0, W, H = _clip_rect(x0, y0, W, H, W_img, H_img)
        roi = img_bgr[y0:y0 + H, x0:x0 + W].copy()
        return roi, (x0, y0, W, H)

    elif mode == "center":
        W = int(w * 0.70)
        H = int(h * 0.28)
        x0 = x + (w - W) // 2
        y0 = y + int(h * 0.42)
        x0, y0, W, H = _clip_rect(x0, y0, W, H, W_img, H_img)
        roi = img_bgr[y0:y0 + H, x0:x0 + W].copy()
        return roi, (x0, y0, W, H)

    elif mode == "dual":
        import numpy as np
        s = int(min(w, h) * 0.22)
        yc = y + int(h * 0.55)
        xl = x + int(w * 0.30) - s // 2
        yl = yc - s // 2
        xr = x + int(w * 0.70) - s // 2
        yr = yc - s // 2
        xl, yl, slw, slh = _clip_rect(xl, yl, s, s, W_img, H_img)
        xr, yr, srw, srh = _clip_rect(xr, yr, s, s, W_img, H_img)
        L = img_bgr[yl:yl + slh, xl:xl + slw]
        R = img_bgr[yr:yr + srh, xr:xr + srh]
        m = min(L.shape[0], R.shape[0])
        n = min(L.shape[1], R.shape[1])
        roi = np.hstack([L[:m, :n], R[:m, :n]])
        return roi, (xl, yl, roi.shape[1], roi.shape[0])

    else:
        raise ValueError("mode must be 'midface'|'center'|'dual'")

def _skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # 노이즈 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask > 0

def roi_to_lab_feature(roi_bgr, use_skin_mask=True, blur_ksize=3):
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("ROI가 비어있습니다.")

    roi = roi_bgr.copy()
    if blur_ksize and blur_ksize % 2 == 1:
        roi = cv2.GaussianBlur(roi, (blur_ksize, blur_ksize), 0)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    if use_skin_mask:
        mask = _skin_mask_ycrcb(roi_bgr)
        cov = float(mask.mean())
        if cov < 0.15:
            Lm, am, bm = float(L.mean()), float(a.mean()), float(b.mean())
            return Lm, am, bm, cov
        Lm = float(L[mask].mean())
        am = float(a[mask].mean())
        bm = float(b[mask].mean())
        return Lm, am, bm, cov
    else:
        return float(L.mean()), float(a.mean()), float(b.mean()), 1.0





