import io
from pathlib import Path
import numpy as np
import cv2, joblib
import streamlit as st
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from src.face_and_color import (
    detect_faces_bboxes, extract_cheek_roi, roi_to_lab_feature
)

# 유틸
def imread_exif_bgr(file_or_bytes):
    if isinstance(file_or_bytes, (str, Path)):
        im = Image.open(file_or_bytes)
    else:
        data = file_or_bytes.read()
        file_or_bytes.seek(0)
        im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.array(im)[:, :, ::-1]
    return arr

@st.cache_resource
def load_payload(model_path: Path):
    payload = joblib.load(model_path)
    pipe = payload["model"]
    feats = payload.get("features", ["a", "b"])
    labels = payload.get("labels", list(pipe.named_steps["svc"].classes_))
    thr = float(payload.get("threshold", 0.5))
    warm_idx = labels.index("warm") if "warm" in labels else 0
    return pipe, feats, labels, thr, warm_idx

def annotate_boxes(img_bgr, face_bbox, roi_rect):
    vis = img_bgr.copy()
    fx, fy, fw, fh = face_bbox
    x0, y0, W, H = roi_rect
    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    cv2.rectangle(vis, (x0, y0), (x0 + W, y0 + H), (255, 0, 0), 2)
    return vis

# UI
st.set_page_config(page_title="퍼스널 컬러 간단 분석", layout="wide")
st.title("퍼스널 컬러 AI 간단 분석")

default_model = "models/svm_ab.joblib"
default_model = Path(default_model)
if not default_model.exists():
    st.error(f"모델이 없습니다: {default_model}")
    st.stop()

pipe, feats, labels, saved_thr, warm_idx = load_payload(default_model)

uploaded = st.file_uploader("사진 업로드 (jpg | jpeg | png )", type=["jpg","jpeg","png"])

if uploaded:
    bgr = imread_exif_bgr(uploaded)
    bboxes = detect_faces_bboxes(bgr)
    if not bboxes:
        st.error("얼굴을 찾지 못했어요. 정면/밝은 사진으로 다시 시도해 주세요.")
        st.stop()

    face = bboxes[0]
    roi, (x0, y0, W, H) = extract_cheek_roi(bgr, face, mode="midface")

    # LAB 추출
    Lm, am, bm, cov = roi_to_lab_feature(roi, use_skin_mask=True)

    # 예측
    if feats == ["a", "b"]:
        X = np.array([[am, bm]], dtype=np.float32)
    elif feats == ["L", "a", "b"]:
        X = np.array([[Lm, am, bm]], dtype=np.float32)
    else:
        st.error(f"지원하지 않는 특징 구성: {feats}")
        st.stop()

    proba = pipe.predict_proba(X)[0]
    p_warm = float(proba[warm_idx])
    pred = "warm" if p_warm >= saved_thr else "cool"
    conf = p_warm if pred == "warm" else (1.0 - p_warm)
    percent = int(round(conf * 100))

    # 시각화(텍스트 없이 박스만)
    vis = annotate_boxes(bgr, face, (x0, y0, W, H))

    # 좌 분석 이미지 / 우 결과
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="분석된 이미지(얼굴/ROI 박스)", use_column_width=True)
    with right:
        is_warm = (pred == "warm")
        label_kor = "웜톤" if is_warm else "쿨톤"
        bg  = "#ffefe6" if is_warm else "#e6f0ff"
        fg  = "#d24b00" if is_warm else "#0b3d91"
        bd  = "#ffb894" if is_warm else "#94b6ff"
        st.markdown(f"""
        <div style="padding:22px;border-radius:18px;background:{bg};border:2px solid {bd};">
            <div style="font-size:54px;font-weight:800;color:{fg};text-align:center;letter-spacing:0.5px;">
                {label_kor.upper()}
            </div>
            <div style="font-size:22px;text-align:center;margin-top:8px;color:#333;">
                정확도 <b>{percent}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("사진을 업로드하면 좌측에 분석 이미지가, 우측에 결과(웜/쿨 · 정확도)가 표시됩니다.")
