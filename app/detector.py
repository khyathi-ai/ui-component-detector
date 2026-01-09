import cv2
import numpy as np

def iou(box1, box2):
    """
    Intersection over Union for normalized boxes
    box = {x, y, w, h}
    """
    x1, y1, w1, h1 = box1.values()
    x2, y2, w2, h2 = box2.values()

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def contains(big, small):
    """Check if big box fully contains small box"""
    return (
        small["x"] >= big["x"] and
        small["y"] >= big["y"] and
        small["x"] + small["w"] <= big["x"] + big["w"] and
        small["y"] + small["h"] <= big["y"] + big["h"]
    )


def classify_ui_component(x, y, w, h, img_w, img_h):
    """
    Classify UI components using layout-based heuristics.
    All rules are relative to image dimensions to ensure consistency.
    """

    rel_w = w / img_w
    rel_h = h / img_h

    # Navigation bar (top horizontal strip)
    if y < img_h * 0.15 and rel_w > 0.6:
        return "navigation_bar"

    # Sidebar (tall vertical strip)
    if rel_h > 0.5 and rel_w < 0.25:
        return "sidebar"

    # Large container / panel
    if rel_w > 0.45 and rel_h > 0.25:
        return "container"

    # Card (medium-sized container)
    if 0.2 < rel_w <= 0.45 and 0.15 < rel_h <= 0.3:
        return "card"

    # Input field (wide but short)
    if rel_w > 0.3 and rel_h < 0.12:
        return "input_field"

    # Button (small rectangle)
    if rel_w < 0.3 and rel_h < 0.15:
        return "button"

    # Icon / badge (very small square-like)
    if rel_w < 0.1 and rel_h < 0.1:
        return "icon"

    # Fallback
    return "ui_section"


def detect_ui_elements(image_bytes: bytes):
    """
    Detect UI elements in a screenshot using OpenCV-based layout heuristics.
    Returns structured JSON with bounding boxes and metadata.
    """

    # Decode image
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    # Original dimensions
    orig_h, orig_w, _ = image.shape

    # Add padding to avoid edge clipping
    pad = int(0.05 * min(orig_w, orig_h))  # 5% padding
    image = cv2.copyMakeBorder(
        image,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # white background
    )

    # Updated dimensions after padding
    img_h, img_w, _ = image.shape


    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    elements = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise / very small regions
        if w < 40 or h < 30:
            continue

        ui_type = classify_ui_component(x, y, w, h, img_w, img_h)

        # Shift box back to original image coordinates
        x_orig = max(0, x - pad)
        y_orig = max(0, y - pad)
        w_orig = min(w, orig_w - x_orig)
        h_orig = min(h, orig_h - y_orig)

        elements.append({
            "type": ui_type,
            "confidence": 0.7,
            "description": f"Detected {ui_type} based on layout heuristics",
            "bounds": {
                "x": round(x_orig / orig_w, 3),
                "y": round(y_orig / orig_h, 3),
                "w": round(w_orig / orig_w, 3),
                "h": round(h_orig / orig_h, 3)
            }
        })

    elements= class_aware_nms(elements)
    return {"elements": elements}

def class_aware_nms(elements, iou_thresh=0.6):
    """
    Suppress overlapping boxes of the SAME class,
    while preserving containment hierarchy.
    """
    kept = []

    for elem in elements:
        suppress = False
        for kept_elem in kept:
            # Only compare same class
            if elem["type"] != kept_elem["type"]:
                continue

            iou_score = iou(elem["bounds"], kept_elem["bounds"])

            # If one contains the other, keep both
            if contains(kept_elem["bounds"], elem["bounds"]) or \
               contains(elem["bounds"], kept_elem["bounds"]):
                continue

            if iou_score > iou_thresh:
                suppress = True
                break

        if not suppress:
            kept.append(elem)

    return kept
