import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Custom modules
from architecture import MyCNN
from gray_scale_conversion import to_grayscale
from histogram_equalization import apply_clahe
from detection_utils import group_overlapping_boxes, remove_contained_boxes, tighten_box


# --- NEW: Selective Search Function ---
def predict_selective_search(model, full_image_np, class_names, conf_threshold=0.9):
    """
    Uses OpenCV Selective Search to propose regions, then classifies them with MyCNN.
    """
    # 1. Initialize Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(full_image_np)

    # "Fast" gives ~2,000 boxes. "Quality" gives ~10,000 (too slow for CPU).
    ss.switchToSelectiveSearchFast()

    # 2. Run Selective Search (returns x, y, w, h)
    rects = ss.process()

    detected_boxes = []
    window_size = (100, 100)  # Input size for MyCNN

    # 3. Batch process regions (optional optimization: batching tensors speeds this up)
    for (x, y, w, h) in rects:
        # Filter out tiny garbage boxes to save time
        if w < 50 or h < 50:
            continue

        # Crop the candidate region
        roi = full_image_np[y:y + h, x:x + w]

        if roi.size == 0: continue

        # --- PREPROCESS FOR MyCNN ---
        # 1. Grayscale
        gray = to_grayscale(roi)
        # 2. CLAHE
        gray_clahe = apply_clahe(gray)
        if gray_clahe.ndim == 3:
            gray_clahe = gray_clahe.squeeze(0)

        # 3. Resize to 100x100
        resized = cv2.resize(gray_clahe, window_size)

        # 4. To Tensor
        tensor_input = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # --- PREDICT ---
        with torch.no_grad():
            outputs = model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, dim=1)

        score = score.item()
        label_idx = predicted_idx.item()

        # Filter by confidence
        if score > conf_threshold:
            # Save in format [x1, y1, x2, y2, score, label]
            detected_boxes.append([x, y, x + w, y + h, score, label_idx])

    return detected_boxes


# --- STREAMLIT UI ---

st.title("Object Detection: Selective Search")
st.write("Using OpenCV Selective Search to find regions + MyCNN to classify them.")

# Configuration Sidebar
st.sidebar.header("Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.4, 1.0, 0.85)
nms_thresh = st.sidebar.slider("Overlap Threshold (NMS)", 0.0, 1.0, 0.3)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load Class Names
try:
    class_csv = "training_data/labels.csv"
    filenames_classnames = np.genfromtxt(class_csv, delimiter=';', skip_header=1, dtype=str)
    class_names = np.unique(filenames_classnames[:, 1])
    class_names.sort()
except Exception:
    class_names = [f"Class_{i}" for i in range(20)]


# Load Model
@st.cache_resource
def load_model():
    model = MyCNN(num_classes=len(class_names))
    try:
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    except FileNotFoundError:
        st.error("Model file 'model.pth' not found.")
        return None
    model.eval()
    return model


model = load_model()

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original Image", width=300)

    if st.button("Detect Objects"):
        with st.spinner("Running Selective Search & Classification..."):

            # Run Detection using the NEW function
            raw_boxes = predict_selective_search(model, img_np, class_names, conf_thresh)

            if len(raw_boxes) > 0:
                # 1. NMS
                nms_boxes = group_overlapping_boxes(np.array(raw_boxes), overlap_thresh=nms_thresh)
                # 2. Containment Filter (from previous step)
                clean_boxes = remove_contained_boxes(nms_boxes)

                # 3. NEW: Tighten the boxes
                final_boxes = []
                for box in clean_boxes:
                    tight_box = tighten_box(img_np, box)
                    final_boxes.append(tight_box)
            else:
                final_boxes = []

            # Draw Boxes
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            st.write(f"Found {len(final_boxes)} objects.")

            for box in final_boxes:
                x1, y1, x2, y2, score, label_idx = box
                label_name = class_names[int(label_idx)]

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                text = f"{label_name}: {score:.2f}"

                # Text Background
                if hasattr(draw, "textbbox"):
                    text_bbox = draw.textbbox((x1, y1), text, font=font)
                    draw.rectangle(text_bbox, fill="red")
                else:
                    w_text, h_text = draw.textsize(text, font=font)
                    draw.rectangle([x1, y1, x1 + w_text, y1 + h_text], fill="red")

                draw.text((x1, y1), text, fill="white", font=font)

            st.image(draw_img, caption="Detected Objects", use_column_width=True)