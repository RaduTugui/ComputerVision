import numpy as np
import cv2  # <--- Added missing import


def sliding_window(image, step_size, window_size):
    """
    Yields (x, y, window) chunks from the image.
    """
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def tighten_box(original_image, box, padding=5):
    """
    Refines a bounding box by finding the object's contours inside the box.
    Works best on images with solid/light backgrounds.
    """
    # <--- FIXED: Ensure coordinates are integers for slicing
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Extract the region of interest (ROI)
    roi = original_image[y1:y2, x1:x2]

    if roi.size == 0:
        return box

    # 1. Convert to grayscale and blur slightly
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thresholding: Detect the object vs background
    # (THRESH_BINARY_INV assumes light background, dark object)
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours of the object inside the ROI
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return box  # Return original if no contour found

    # 4. Find the largest contour (the object)
    c = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(c)

    # 5. Calculate new coordinates relative to the ORIGINAL image
    new_x1 = x1 + rx - padding
    new_y1 = y1 + ry - padding
    new_x2 = x1 + rx + rw + padding
    new_y2 = y1 + ry + rh + padding

    # Ensure we don't go out of image bounds
    h_img, w_img, _ = original_image.shape
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w_img, new_x2)
    new_y2 = min(h_img, new_y2)

    # Return updated box with original score/label
    return [new_x1, new_y1, new_x2, new_y2, box[4], box[5]]


def remove_contained_boxes(boxes):
    """
    Removes boxes that are completely (or mostly) inside another box with higher confidence.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by confidence (highest first)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    keep = []

    for i, current_box in enumerate(boxes):
        cx1, cy1, cx2, cy2 = current_box[:4]
        current_area = (cx2 - cx1) * (cy2 - cy1)
        is_contained = False

        for kept_box in keep:
            kx1, ky1, kx2, ky2 = kept_box[:4]

            # Check if current_box is inside kept_box
            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection_area = iw * ih

            # If >90% of the current box is inside a bigger box
            if intersection_area > 0.90 * current_area:
                is_contained = True
                break

        if not is_contained:
            keep.append(current_box)

    return keep


def group_overlapping_boxes(boxes, overlap_thresh=0.1):
    """
    ADVANCED: Instead of deleting overlaps, it AVERAGES them.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    final_boxes = []

    while len(boxes) > 0:
        base_box = boxes[0]

        x1 = np.maximum(base_box[0], boxes[:, 0])
        y1 = np.maximum(base_box[1], boxes[:, 1])
        x2 = np.minimum(base_box[2], boxes[:, 2])
        y2 = np.minimum(base_box[3], boxes[:, 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        intersection = w * h

        area_base = (base_box[2] - base_box[0]) * (base_box[3] - base_box[1])
        area_others = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iou = intersection / (area_base + area_others - intersection)

        overlapping_indices = np.where(iou > overlap_thresh)[0]
        subset = boxes[overlapping_indices]

        # Average coordinates
        avg_box = np.mean(subset, axis=0)

        # Keep max score and mode label
        avg_box[4] = np.max(subset[:, 4])
        labels = subset[:, 5].astype(int)
        avg_box[5] = np.bincount(labels).argmax()

        final_boxes.append(avg_box)
        boxes = np.delete(boxes, overlapping_indices, axis=0)

    return np.array(final_boxes)