import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Custom modules
from architecture import MyCNN
from gray_scale_conversion import to_grayscale
from histogram_equalization import apply_clahe
# Ensure you have the detection_utils.py file created previously
from detection_utils import sliding_window, non_max_suppression, group_overlapping_boxes


def predict_sliding_window(model, full_image_np, class_names, step_size=20, conf_threshold=0.9):
    """
    Scans the image at MULTIPLE SCALES (Image Pyramid) to find objects of different sizes.
    """

    # 1. Preprocess FULL image
    gray = to_grayscale(full_image_np)
    gray_clahe = apply_clahe(gray)

    # Ensure (H, W) format for OpenCV resizing
    if gray_clahe.ndim == 3:
        gray_clahe = gray_clahe.squeeze(0)

    window_size = (100, 100)  # Must match model training input
    detected_boxes = []

    # --- IMAGE PYRAMID CONFIGURATION ---
    # Scales: 1.0 (finding small objects) -> 0.15 (finding huge objects)
    scales = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]

    for scale in scales:
        # Resize the image for this iteration
        scaled_h = int(gray_clahe.shape[0] * scale)
        scaled_w = int(gray_clahe.shape[1] * scale)

        # Stop if image is smaller than the window
        if scaled_h < window_size[1] or scaled_w < window_size[0]:
            continue

        scaled_img = cv2.resize(gray_clahe, (scaled_w, scaled_h))

        # Dynamic step size: smaller steps for smaller scales to be precise
        current_step = int(step_size * scale)
        if current_step < 10: current_step = 10

        # 2. Slide window on the SCALED image
        for (x, y, window) in sliding_window(scaled_img, current_step, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            tensor_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                score, predicted_idx = torch.max(probabilities, dim=1)

            score = score.item()
            label_idx = predicted_idx.item()

            if score > conf_threshold:
                # 3. UPSCALING COORDINATES
                # Map the box from the small/scaled image back to the original image
                orig_x1 = int(x / scale)
                orig_y1 = int(y / scale)
                orig_x2 = int((x + window_size[0]) / scale)
                orig_y2 = int((y + window_size[1]) / scale)

                detected_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2, score, label_idx])

    return detected_boxes


# --- STREAMLIT UI ---

st.title("Object Detection via Classification")
st.write("Upload an image. The model will scan for objects using a Sliding Window.")

# Configuration Sidebar
st.sidebar.header("Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.4, 1.0, 0.70)
nms_thresh = st.sidebar.slider("Overlap Threshold (NMS)", 0.0, 1.0, 0.2)
step_size = st.sidebar.slider("Step Size (Pixels)", 10, 100, 20, help="Lower is slower but more accurate")

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
    # Load and show original
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original Image", width=300)

    if st.button("Detect Objects"):
        with st.spinner("Scanning image at multiple scales (this may take a moment)..."):

            #Run Detection
            raw_boxes = predict_sliding_window(model, img_np, class_names, step_size, conf_thresh)

            if len(raw_boxes) > 0:
                final_boxes = group_overlapping_boxes(np.array(raw_boxes), overlap_thresh=nms_thresh)
            else:
                final_boxes = []

            # 3. Draw Boxes
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)

            # Optional: Load a font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            st.write(f"Found {len(final_boxes)} objects.")

            for box in final_boxes:
                x1, y1, x2, y2, score, label_idx = box
                label_name = class_names[int(label_idx)]

                # Draw Box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Draw Label
                text = f"{label_name}: {score:.2f}"

                # Draw text background
                if hasattr(draw, "textbbox"):
                    text_bbox = draw.textbbox((x1, y1), text, font=font)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
                else:
                    w, h = draw.textsize(text, font=font)
                    draw.rectangle([x1, y1, x1 + w, y1 + h], fill="red")

                draw.text((x1, y1), text, fill="white", font=font)

            st.image(draw_img, caption="Detected Objects", use_column_width=True)

# import streamlit as st
# import torch
# import numpy as np
# import cv2
# from PIL import Image, ImageDraw, ImageFont
# from ultralytics import YOLO
# from architecture import MyCNN
# from gray_scale_conversion import to_grayscale
# from histogram_equalization import apply_clahe
#
#
# # --- HELPER: Preprocess Crop for MyCNN ---
# def preprocess_crop_for_mycnn(crop_img_np):
#     """
#     Takes an RGB crop (H, W, 3), converts to Grayscale -> CLAHE -> Resize 100x100
#     Returns tensor (1, 1, 100, 100)
#     """
#     # 1. To Grayscale
#     gray = to_grayscale(crop_img_np)
#
#     # 2. CLAHE
#     gray_clahe = apply_clahe(gray)
#     if gray_clahe.ndim == 3:
#         gray_clahe = gray_clahe.squeeze(0)  # Remove channel dim to get (H, W)
#
#     # 3. Resize to 100x100 (Input size for MyCNN)
#     # Note: For best results, you might want to pad to square first,
#     # but direct resize often works well enough for CNNs.
#     resized = cv2.resize(gray_clahe, (100, 100))
#
#     # 4. To Tensor
#     tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     return tensor
#
#
# # --- PREDICTION LOGIC: HYBRID YOLO + MyCNN ---
# def predict_with_yolo_hybrid(my_model, yolo_model, full_image_np, class_names, conf_threshold=0.5):
#     """
#     1. Uses YOLO to find bounding boxes.
#     2. Uses MyCNN to classify the content of those boxes.
#     """
#     detected_boxes = []
#
#     # 1. Run YOLO inference (Generic Object Detection)
#     # conf=0.2 detects even faint objects. agnostic_nms=True prevents overlapping boxes.
#     results = yolo_model(full_image_np, conf=0.2, verbose=False, agnostic_nms=True)
#
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             # Get integer coordinates
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#             # Ensure coordinates are within image bounds
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(full_image_np.shape[1], x2), min(full_image_np.shape[0], y2)
#
#             # 2. CROP the object
#             crop = full_image_np[y1:y2, x1:x2]
#
#             if crop.size == 0: continue
#
#             # 3. Classify with MyCNN
#             tensor_input = preprocess_crop_for_mycnn(crop)
#
#             with torch.no_grad():
#                 outputs = my_model(tensor_input)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 score, predicted_idx = torch.max(probabilities, dim=1)
#
#             score = score.item()
#             label_idx = predicted_idx.item()
#
#             # 4. Filter by MyCNN confidence (NOT YOLO confidence)
#             if score > conf_threshold:
#                 detected_boxes.append([x1, y1, x2, y2, score, label_idx])
#
#     return detected_boxes
#
#
# # --- STREAMLIT UI ---
# st.title("Hybrid Detector: YOLO + MyCNN")
# st.write("Using YOLO to find objects & MyCNN to classify them.")
#
# # Sidebar
# st.sidebar.header("Settings")
# conf_thresh = st.sidebar.slider("MyCNN Confidence", 0.0, 1.0, 0.60)
#
#
# # Load Models
# @st.cache_resource
# def load_models():
#     # Load your Custom Classifier
#     my_cnn = MyCNN(num_classes=20)  # Update 20 if you have a different count
#     try:
#         my_cnn.load_state_dict(torch.load("model.pth", map_location="cpu"))
#         my_cnn.eval()
#     except:
#         st.error("Could not load model.pth")
#
#     # Load Standard YOLOv8 Nano (Small & Fast)
#     # It will download 'yolov8n.pt' automatically on first run
#     yolo_net = YOLO('yolov8n.pt')
#
#     return my_cnn, yolo_net
#
#
# # Load Class Names
# try:
#     class_csv = "training_data/labels.csv"
#     filenames_classnames = np.genfromtxt(class_csv, delimiter=';', skip_header=1, dtype=str)
#     class_names = np.unique(filenames_classnames[:, 1])
#     class_names.sort()
# except Exception:
#     class_names = [f"Class_{i}" for i in range(20)]
#
# my_model, yolo_model = load_models()
#
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
#
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     img_np = np.array(image)
#
#     st.image(image, caption="Original", width=300)
#
#     if st.button("Detect Objects"):
#         with st.spinner("YOLO is finding objects, MyCNN is classifying them..."):
#
#             # Run Hybrid Detection
#             final_boxes = predict_with_yolo_hybrid(my_model, yolo_model, img_np, class_names, conf_thresh)
#
#             # Draw
#             draw_img = image.copy()
#             draw = ImageDraw.Draw(draw_img)
#             try:
#                 font = ImageFont.truetype("arial.ttf", 20)
#             except:
#                 font = ImageFont.load_default()
#
#             st.write(f"Found {len(final_boxes)} objects.")
#
#             for box in final_boxes:
#                 x1, y1, x2, y2, score, label_idx = box
#                 label_name = class_names[int(label_idx)]
#
#                 # Draw Box
#                 draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
#
#                 # Draw Label
#                 label_text = f"{label_name} ({score:.2f})"
#
#                 # Text Background
#                 if hasattr(draw, "textbbox"):
#                     bbox = draw.textbbox((x1, y1), label_text, font=font)
#                     draw.rectangle(bbox, fill="lime")
#                 else:
#                     w, h = draw.textsize(label_text, font=font)
#                     draw.rectangle([x1, y1, x1 + w, y1 + h], fill="lime")
#
#                 draw.text((x1, y1), label_text, fill="black", font=font)
#
#             st.image(draw_img, caption="Hybrid Detection Results", use_column_width=True)