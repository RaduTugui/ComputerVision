import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

def apply_watershed(image_np: np.ndarray) -> np.ndarray:
    """
    Applies Otsu's thresholding, Distance Transform, and Watershed segmentation.
    Expects input shape (1, H, W) or (H, W).
    Returns shape (1, H, W) with values normalized 0-255.
    """
    # 1. Ensure input is 2D for OpenCV processing
    # The 'to_grayscale' function usually returns (1, H, W).
    # We need (H, W) for cv2.
    if image_np.ndim == 3:
        gray = image_np[0]
    else:
        gray = image_np

    # Ensure it's 8-bit uint8 (0-255) for OpenCV
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # 2. Otsu's Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Distance Transform
    distance = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # 4. Find Peaks (Markers)
    distance_norm = cv2.normalize(distance, None, 0, 1.0, cv2.NORM_MINMAX)
    # Using 0.4 as threshold to find sure-foreground areas
    local_max = distance_norm > 0.4 * distance_norm.max()
    markers, _ = ndi.label(local_max)

    # 5. Apply Watershed
    labels = watershed(-distance, markers, mask=thresh.astype(bool))

    # 6. Normalize Labels to be valid image data (0-255)
    # This ensures the CNN sees consistent values
    if labels.max() > 0:
        processed = (labels / labels.max()) * 255
    else:
        processed = labels

    # 7. Restore shape to (1, H, W)
    # This is crucial so it works with the rest of your pipeline
    return processed.astype(image_np.dtype)[None, :, :]