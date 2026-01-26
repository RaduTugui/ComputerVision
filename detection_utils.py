import numpy as np


def sliding_window(image, step_size, window_size):
    """
    Yields (x, y, window) chunks from the image.
    """
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    Standard NMS: Picks the single highest scoring box and deletes neighbors.
    Good for removing duplicates, but doesn't fix centering.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick]


def group_overlapping_boxes(boxes, overlap_thresh=0.1):
    """
    ADVANCED: Instead of deleting overlaps, it AVERAGES them.
    This creates a new box that is the average center of all detections.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    final_boxes = []

    while len(boxes) > 0:
        # Take the first box
        base_box = boxes[0]

        # Find all boxes that overlap significantly with this base box
        x1 = np.maximum(base_box[0], boxes[:, 0])
        y1 = np.maximum(base_box[1], boxes[:, 1])
        x2 = np.minimum(base_box[2], boxes[:, 2])
        y2 = np.minimum(base_box[3], boxes[:, 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        intersection = w * h

        area_base = (base_box[2] - base_box[0]) * (base_box[3] - base_box[1])
        area_others = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # IoU (Intersection over Union)
        iou = intersection / (area_base + area_others - intersection)

        # Get indices of overlapping boxes
        overlapping_indices = np.where(iou > overlap_thresh)[0]

        # Average their coordinates
        subset = boxes[overlapping_indices]
        avg_box = np.mean(subset, axis=0)

        # Keep the max score, but averaged coordinates
        avg_box[4] = np.max(subset[:, 4])
        # Keep the most common label (mode)
        labels = subset[:, 5].astype(int)
        avg_box[5] = np.bincount(labels).argmax()

        final_boxes.append(avg_box)

        # Remove processed boxes
        boxes = np.delete(boxes, overlapping_indices, axis=0)

    return np.array(final_boxes)