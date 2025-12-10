import cv2
import numpy as np

def apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Expects input shape (1, H, W) or (H, W), typically after grayscale conversion.
    Returns shape (1, H, W) with enhanced contrast.
    """
    # Ensure input is 2D and 8-bit uint8 for OpenCV
    if image_np.ndim == 3:
        # Assuming format (1, H, W) from to_grayscale
        gray = image_np[0]
    else:
        gray = image_np

    if gray.dtype != np.uint8:
        # Convert to 0-255 range
        gray = gray.astype(np.uint8)

    # create CLAHE object
    # clipLimit controls the contrast enhancement limit
    # tileGridSize defines the size of the blocks where histogram equalization is applied
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply CLAHE
    enhanced_image = clahe.apply(gray)

    # Restore shape to (1, H, W) to match pipeline expectations
    return enhanced_image[None, :, :]