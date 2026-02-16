import numpy as np
import cv2


# Change std_dev_thresh from 15 to 5
def is_box_valid(roi, std_dev_thresh=5, min_aspect=0.3, max_aspect=3.0):
    """
    SMART FILTER: Rejects boxes that don't look like real objects.
    """
    if roi.size == 0: return False
    h, w = roi.shape[:2]

    # Aspect Ratio
    ratio = w / float(h)
    if ratio < min_aspect or ratio > max_aspect:
        return False

    # Texture
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi

    mean, std_dev = cv2.meanStdDev(gray)


    if std_dev[0][0] < std_dev_thresh:
        return False

    return True

def group_overlapping_boxes(boxes, overlap_thresh=0.15):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    final_boxes = []

    # Get unique classes detected (e.g., 0=Cat, 1=Dog)
    unique_classes = np.unique(boxes[:, 5])

    for cls in unique_classes:
        # Get only boxes for this specific class
        cls_indices = np.where(boxes[:, 5] == cls)[0]
        cls_boxes = boxes[cls_indices]

        pick = []
        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 2]
        y2 = cls_boxes[:, 3]
        scores = cls_boxes[:, 4]
        area = (x2 - x1) * (y2 - y1)

        idxs = np.argsort(scores)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (area[i] + area[idxs[:last]] - intersection)

            # Delete highly overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlap_thresh)[0])))

        final_boxes.extend(cls_boxes[pick])

    return np.array(final_boxes)


def remove_contained_boxes(boxes):
    """Removes a box if it is completely inside another box of the same class."""
    if len(boxes) == 0: return []

    # Sort by size (area), largest first
    boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    keep = []

    for current_box in boxes:
        is_contained = False
        cx1, cy1, cx2, cy2 = current_box[:4]
        c_label = current_box[5]
        c_area = (cx2 - cx1) * (cy2 - cy1)

        for kept_box in keep:
            kx1, ky1, kx2, ky2 = kept_box[:4]
            k_label = kept_box[5]

            # Only remove if it's contained in a box of the SAME class
            # (Don't let a big Chair delete a small Cat)
            if c_label != k_label:
                continue

            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            # If 90% contained, delete it
            if intersection > 0.90 * c_area:
                is_contained = True
                break

        if not is_contained:
            keep.append(current_box)

    return keep


def tighten_box(original_image, box, padding=5):
    """
    Tries to shrink the box to fit the object contour.
    Warning: Can be unstable on complex backgrounds.
    """
    h_img, w_img, _ = original_image.shape
    x1, y1, x2, y2 = map(int, box[:4])

    # Safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    roi = original_image[y1:y2, x1:x2]
    if roi.size == 0: return box

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return box

    c = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(c)

    return [
        max(0, x1 + rx - padding),
        max(0, y1 + ry - padding),
        min(w_img, x1 + rx + rw + padding),
        min(h_img, y1 + ry + rh + padding),
        box[4], box[5]
    ]