import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from architecture import MyCNN
from prepare_images import prepare_image
from watershed_transform import apply_watershed
from gray_scale_conversion import to_grayscale


def predict_single_image(image_path, model_path="model.pth", class_csv="training_data/labels.csv"):
    # 1. Setup Device
    # Use GPU if available, as the model was trained with CUDA/AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning inference on: {device}")

    # 2. Load the Class Names
    try:
        # NOTE: Using the correct path: training_data/labels.csv
        filenames_classnames = np.genfromtxt(class_csv, delimiter=';', skip_header=1, dtype=str)
        class_names = np.unique(filenames_classnames[:, 1])
        class_names.sort()
    except Exception:
        class_names = None
        print(f"Warning: Could not load class names from {class_csv}")

    # 3. Load the Model
    model = MyCNN(num_classes=20)
    # Map the model to the GPU if available
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Load and Process the Image
    print(f"Processing: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    with Image.open(image_path) as im:
        image_np = np.array(im, dtype=np.uint8)

    # Convert to grayscale → shape should be (1, H, W)
    image_gs = to_grayscale(image_np)

    # Apply watershed → shape (1, H, W)
    processed_image = apply_watershed(image_gs)

    # Resize to 100×100 → shape (1, 100, 100)
    # FIX: Using 100 for size as per the final Streamlit app fix
    resized_image, _ = prepare_image(processed_image, 100, 100, 0, 0, 100)

    # Enforce correct shape
    if resized_image.ndim == 2:
        resized_image = resized_image[np.newaxis, :, :]
    elif resized_image.shape[0] != 1:
        resized_image = resized_image[:1]  # force single channel

    # 5. Convert to Tensor
    image_tensor = torch.tensor(resized_image, dtype=torch.float32) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # shape is (1, 1, 100, 100)
    image_tensor = image_tensor.to(device)

    # 6. Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    # 7. Prepare text (MODIFIED FOR REQUEST)
    if class_names is not None:
        predicted_label = class_names[predicted_idx]
        # Successful lookup: print the label name
        prediction_text = f"Label: {predicted_label} ({confidence * 100:.2f}%)"
    else:
        # Failed lookup: print "Label: ID XX" as requested
        prediction_text = f"Label: ID {predicted_idx} ({confidence * 100:.2f}%)"

    print(f"\nPrediction → {prediction_text}\n")

    # Visualization
    plt.imshow(resized_image[0], cmap='gray')
    plt.title(f"Prediction: {prediction_text}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Ensure this path is correct for your local machine
    test_img = "/home/radu/Downloads/cat.jpg"

    # NOTE: Your CSV file is named 'labels.csv' and is in 'training_data'
    predict_single_image(test_img, class_csv="training_data/labels.csv")