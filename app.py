import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


from architecture import MyCNN
from gray_scale_conversion import to_grayscale
from histogram_equalization import apply_clahe
from detection_utils import group_overlapping_boxes, remove_contained_boxes, is_box_valid

CONFIDENCE_THRESHOLD = 0.85
NMS_THRESHOLD = 0.10
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5
STD_DEV_THRESHOLD = 20


#Loading the labels
@st.cache_data
def load_class_names(csv_path="training_data/labels.csv"):
    try:
        # Load CSV with ';' delimiter as seen in your file
        df = pd.read_csv(csv_path, sep=';')
        # Get unique labels and sort them (to match training order)
        class_names = sorted(df['label'].unique())
        return class_names
    except Exception as e:
        st.error(f"Error loading labels.csv: {e}")
        return [f"Class_{i}" for i in range(20)]



def predict_and_count(model, full_image_np, class_names):
    # Initialize Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(full_image_np)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    batch_tensors = []
    batch_coords = []
    window_size = (100, 100)

    # Limit to 2000 boxes for speed
    if len(rects) > 2000:
        rects = rects[:2000]

    for (x, y, w, h) in rects:
        #Filter tiny boxes
        if w < 50 or h < 50: continue

        #Smart Filter: Reject weird shapes (lines) or empty backgrounds
        roi = full_image_np[y:y + h, x:x + w]
        if roi.size == 0: continue

        if not is_box_valid(roi, std_dev_thresh=STD_DEV_THRESHOLD,
                            min_aspect=MIN_ASPECT_RATIO, max_aspect=MAX_ASPECT_RATIO):
            continue

        #Preprocess (Gray -> CLAHE -> Resize)
        gray = to_grayscale(roi)
        gray_clahe = apply_clahe(gray)
        if gray_clahe.ndim == 3: gray_clahe = gray_clahe.squeeze(0)
        resized = cv2.resize(gray_clahe, window_size)

        #Prepare Tensor
        t_roi = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)
        batch_tensors.append(t_roi)
        batch_coords.append((x, y, w, h))

    if not batch_tensors: return [], {}

    #Batch Predict (Run Model ONCE)
    batch_input = torch.stack(batch_tensors)

    with torch.no_grad():
        outputs = model(batch_input)
        probabilities = torch.softmax(outputs, dim=1)
        max_scores, predicted_indices = torch.max(probabilities, dim=1)

    #Collect Valid Detections
    raw_boxes = []
    for i in range(len(batch_tensors)):
        score = max_scores[i].item()
        if score > CONFIDENCE_THRESHOLD:
            idx = predicted_indices[i].item()
            x, y, w, h = batch_coords[i]
            raw_boxes.append([x, y, x + w, y + h, score, idx])

    #Apply NMS (Remove duplicates)
    if len(raw_boxes) > 0:
        nms_boxes = group_overlapping_boxes(raw_boxes, overlap_thresh=NMS_THRESHOLD)
        final_boxes = remove_contained_boxes(nms_boxes)
    else:
        final_boxes = []

    #Count Objects
    counts = {}
    for box in final_boxes:
        label_idx = int(box[5])
        name = class_names[label_idx]
        counts[name] = counts.get(name, 0) + 1

    return final_boxes, counts


# STREAMLIT UI
st.title("Object Detection & Counting")

# Load Labels
class_names = load_class_names()
st.sidebar.success(f"Loaded {len(class_names)} classes: {', '.join(class_names[:5])}...")


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
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Count & Detect"):
        with st.spinner("Analyzing..."):
            final_boxes, counts = predict_and_count(model, img_np, class_names)


            if counts:
                st.success(f"Found {len(final_boxes)} objects!")
                cols = st.columns(3)
                for i, (name, count) in enumerate(counts.items()):
                    cols[i % 3].metric(label=name.capitalize(), value=count)
            else:
                st.warning("No objects found (Try lowering the confidence threshold).")

            #Draw Boxes
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            for box in final_boxes:
                x1, y1, x2, y2, score, label_idx = box
                label_name = class_names[int(label_idx)]

                # Draw Red Box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

                # Draw Label with Background
                text = f"{label_name}: {score:.2f}"
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((x1, y1), text, font=font)
                    draw.rectangle(bbox, fill="red")
                else:
                    w_text, h_text = draw.textsize(text, font=font)
                    draw.rectangle([x1, y1, x1 + w_text, y1 + h_text], fill="red")
                draw.text((x1, y1), text, fill="white", font=font)

            st.image(draw_img, caption="Detected Objects", use_container_width=True)