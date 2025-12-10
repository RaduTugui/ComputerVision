import streamlit as st
import torch
import numpy as np
from PIL import Image

from architecture import MyCNN
from gray_scale_conversion import to_grayscale
from histogram_equalization import apply_clahe
from prepare_images import prepare_image


def preprocess_external_image(image: Image.Image):
    """
    1. Convert to numpy
    2. Grayscale
    3. Contrast Enhancement (CLAHE)
    4. padded resize
    """

    img_np = np.array(image, dtype=np.uint8)

    #grayscale
    gray = to_grayscale(img_np)

    # Insert CLAHE for robust contrast enhancement
    clahe_enhanced_img = apply_clahe(gray)

    # The final image fed into prepare_image is now the CLAHE-enhanced image
    final_img, _ = prepare_image(clahe_enhanced_img, 100, 100, 0, 0, 32)

    # Convert to tensor
    tensor = torch.tensor(final_img, dtype=torch.float32).unsqueeze(0)  # → (1,1,100,100)

    return tensor, final_img

def predict_single_image(model, tensor, class_names):
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return class_names[predicted_idx.item()], confidence.item()

# STREAMLIT UI
st.title("CNN Image Classifier")
st.write("Upload an image and see the prediction!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# Load class names
try:
    class_csv = "training_data/labels.csv"
    filenames_classnames = np.genfromtxt(class_csv, delimiter=';', skip_header=1, dtype=str)
    class_names = np.unique(filenames_classnames[:, 1])
    class_names.sort()
except Exception:
    class_names = [f"Class_{i}" for i in range(20)]


# Load model
try:
    model = MyCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()




# Upload image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            tensor, processed_img = preprocess_external_image(image)
            label, confidence = predict_single_image(model, tensor, class_names)

        st.success(f"Prediction: **{label}** ({confidence * 100:.2f}%)")

        st.write("### Processed Input to Model ")
        st.image(processed_img[0], width=200, clamp=True)
