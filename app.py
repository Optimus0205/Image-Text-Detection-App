import io
import base64
import cv2
import numpy as np
import streamlit as st
from inference import predict

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Image Prediction",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------
# Custom CSS - kept minimal to match the plain mockup design
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #e5e7eb;
    }

    /* Title */
    .app-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        color: #4f46e5;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Shrink and center the file uploader */
    div[data-testid="stFileUploader"] {
        max-width: 380px;
        margin: 0 auto 1rem auto;
    }
    div[data-testid="stFileUploaderDropzone"] {
        padding: 10px;
        border: 1px solid #cfd3da;
        border-radius: 6px;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0;
        display: flex;
        justify-content: center;
    }
    div[data-testid="stFileUploader"] button {
        margin: 0 auto;
    }

    /* Predict button */
    div.stButton {
        max-width: 380px;
        margin: 0 auto 2rem auto;
    }
    div.stButton > button {
        background: #4338ca;
        color: #fff;
        border: none;
        padding: 0.5rem 2.2rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 6px;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #372fa8;
        color: #fff;
        border: none;
    }

    /* Panel headers */
    .panel-header {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.8rem;
    }

    /* Square image box (placeholder + filled state look identical) */
    .image-box {
        width: 100%;
        aspect-ratio: 7 / 10;
        background: #d1d5db;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .image-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .image-box span {
        color: #6b7280;
        font-size: 1.05rem;
        font-weight: 500;
    }

    /* Detections list */
    .detections-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-top: 2rem;
        border: 1px solid #e5e7eb;
    }
    .detection-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #f5f5fd;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .cls-name { font-weight: 700; color: #4338ca; }
    .coords { color: #6b7280; font-size: 0.75rem; }
    .conf-badge {
        font-weight: 600;
        color: #059669;
        background: #ecfdf5;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.markdown('<div class="app-title">Image Prediction</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Upload + Predict (centered, compact)
# ---------------------------------------------------------
_, mid, _ = st.columns([1, 2, 1])
with mid:
    uploaded_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed"
    )
    predict_clicked = st.button("Predict")

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "predicted_b64" not in st.session_state:
    st.session_state.predicted_b64 = None
if "detections" not in st.session_state:
    st.session_state.detections = None

if uploaded_file is not None and st.session_state.get("last_filename") != uploaded_file.name:
    st.session_state.predicted_b64 = None
    st.session_state.detections = None
    st.session_state.last_filename = uploaded_file.name

# ---------------------------------------------------------
# Run prediction (reuses inference.py's predict() as-is)
# ---------------------------------------------------------
if predict_clicked:
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        with st.spinner("Running inference..."):
            try:
                file_bytes = uploaded_file.getvalue()
                image_buffer = io.BytesIO(file_bytes)

                annotated_bytes, detection_results = predict(image_buffer)

                st.session_state.predicted_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
                st.session_state.detections = detection_results
            except Exception as e:
                st.error(f"Inference failed: {e}")

# ---------------------------------------------------------
# Original / Predicted image boxes
# ---------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="panel-header">Original Image</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        original_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        st.markdown(
            f'<div class="image-box"><img src="data:image/jpeg;base64,{original_b64}"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="image-box"><span>Original Image</span></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-header">Predicted Image</div>', unsafe_allow_html=True)
    if st.session_state.predicted_b64 is not None:
        st.markdown(
            f'<div class="image-box"><img src="data:image/jpeg;base64,{st.session_state.predicted_b64}"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="image-box"><span>Predicted Image</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Detections list
# ---------------------------------------------------------
if st.session_state.detections is not None:
    st.markdown('<div class="detections-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header" style="text-align:left;">Detections</div>', unsafe_allow_html=True)

    if len(st.session_state.detections) == 0:
        st.markdown('<div style="text-align:center;color:#6b7280;">No objects detected.</div>', unsafe_allow_html=True)
    else:
        for det in st.session_state.detections:
            box_str = ", ".join(str(c) for c in det["box"])
            st.markdown(
                f"""
                <div class="detection-row">
                    <span class="cls-name">{det['class']}</span>
                    <span class="coords">[{box_str}]</span>
                    <span class="conf-badge">{det['confidence'] * 100:.0f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)