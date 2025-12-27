import streamlit as st
import io
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Car Damage Detection – YOLOv12",
    layout="wide"
)

st.title("Car Damage Detection (YOLOv12)")

# ----------------------------
# Load YOLO Model (ONCE)
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "model.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# ----------------------------
# Inference Function
# ----------------------------
def run_inference(model, img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)

    annotated = results[0].plot()
    annotated_img = Image.fromarray(annotated[:, :, ::-1])

    detections = []
    h, w = results[0].orig_shape

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        area = width * height

        severity = round((conf * 0.7 + (area / (h * w)) * 0.3) * 100, 2)

        detections.append({
            "class": cls_name,
            "confidence": round(conf * 100, 2),
            "severity": severity,
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "width": round(width, 1),
                "height": round(height, 1),
                "area": round(area, 1)
            }
        })

    return annotated_img, detections

# ----------------------------
# File Upload (Image Only)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)

    if st.button("Run Image Detection"):
        with st.spinner("Running image detection..."):
            annotated_img, detections = run_inference(
                model, uploaded_file.getvalue()
            )

        with col2:
            st.subheader("Processed Image")
            st.image(annotated_img, use_container_width=True)

        st.subheader("📊 Detection Summary")

        if not detections:
            st.success("No damage detected.")
        else:
            for i, det in enumerate(detections, 1):
                with st.expander(f"Damage {i}: {det['class']}"):
                    st.write(f"**Confidence:** {det['confidence']}%")
                    st.write(f"**Severity:** {det['severity']} / 100")

                    bbox = det["bbox"]
                    st.write(
                        f"**BBox:** ({bbox['x1']},{bbox['y1']}) → ({bbox['x2']},{bbox['y2']})"
                    )
                    st.write(
                        f"**Size:** {bbox['width']} × {bbox['height']} px"
                    )
                    st.write(f"**Area:** {bbox['area']} px²")

else:
    st.info("⬆ Upload an image to begin")
