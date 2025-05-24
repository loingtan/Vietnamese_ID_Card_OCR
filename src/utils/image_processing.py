"""
Image processing utilities for Vietnamese ID Card OCR.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import io
from qreader import QReader
import torch
from ensemble_boxes import weighted_boxes_fusion


def apply_nms(boxes: np.ndarray, scores: Optional[np.ndarray] = None, nms_thresh: float = 0.3) -> List:
    """
    Apply Non-Maximum Suppression to eliminate overlapping boxes.

    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores (optional)
        nms_thresh: NMS threshold

    Returns:
        List of filtered boxes
    """
    if len(boxes) == 0:
        return []

    # If no scores provided, assume all boxes have equal confidence
    if scores is None:
        scores = np.ones(len(boxes))

    EXPAND = 3
    expanded_boxes = boxes.copy()
    for i, box in enumerate(expanded_boxes):
        x1, y1, x2, y2 = box
        expanded_boxes[i] = [max(0, x1-EXPAND), max(0, y1-EXPAND),
                             x2+EXPAND, y2+EXPAND]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        expanded_boxes.tolist(), scores.tolist(), 0.5, nms_thresh)

    # Return filtered boxes
    filtered_boxes = []
    if len(indices) > 0:
        for idx in indices:
            # OpenCV 4.5.4+ returns a 1D array
            if isinstance(idx, np.ndarray):
                idx = idx.item()
            filtered_boxes.append(boxes[idx])

    return filtered_boxes


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: First box as ((x1, y1), (x2, y2))
        box2: Second box as ((x1, y1), (x2, y2))

    Returns:
        IoU value between 0 and 1
    """
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]

    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def warp_and_recognize(frame: np.ndarray, top_left: Tuple, top_right: Tuple,
                       bottom_right: Tuple, bottom_left: Tuple,
                       vietocr_model) -> List:
    """
    Warp the detected ID card region and perform OCR recognition.

    Args:
        frame: Input image
        top_left, top_right, bottom_right, bottom_left: Corner coordinates
        vietocr_model: VietOCR model for text recognition

    Returns:
        List containing recognized text and box coordinates
    """
    h, w, cn = frame.shape
    padding = 4.0
    padding = int(padding * w / 640)

    # All points are in format [cols, rows]
    pt_A = top_left[0] - padding, top_left[1] - padding
    pt_B = bottom_left[0] - padding, bottom_left[1] + padding
    pt_C = bottom_right[0] + padding, bottom_right[1] + padding
    pt_D = top_right[0] + padding, top_right[1] - padding

    # Calculate dimensions using L2 norm
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    max_width = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) +
                        ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) +
                        ((pt_C[1] - pt_D[1]) ** 2))
    max_height = max(int(height_AB), int(height_CD))

    # Define transformation points
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, max_height - 1],
                            [max_width - 1, max_height - 1],
                            [max_width - 1, 0]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    mat_warped = cv2.warpPerspective(
        frame, M, (max_width, max_height), flags=cv2.INTER_LINEAR)

    # Perform OCR using VietOCR
    recognized_text = vietocr_model.predict(Image.fromarray(mat_warped))

    # Create box coordinates
    box = [pt_A, pt_D, pt_C, pt_B]

    return [recognized_text, box]


def pil_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Image as bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def resize_image(image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply basic image enhancement for better OCR results.

    Args:
        image: Input image

    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Convert back to BGR if original was color
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    else:
        enhanced = blurred

    return enhanced


def corner_preprocess_image(image, device):
    """Resizes an image to fit within 640x640 while maintaining aspect ratio, then pads it.
    Returns the preprocessed image tensor and scaling factors."""
    h, w, _ = image.shape
    scale = 640 / max(h, w)  # Scale factor to fit within 640x640
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping aspect ratio
    image_resized = cv2.resize(image, (new_w, new_h))

    # Create a blank 640x640 canvas (black padding)
    padded_image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Center the resized image
    start_x = (640 - new_w) // 2
    start_y = (640 - new_h) // 2
    padded_image[start_y:start_y + new_h,
                 start_x:start_x + new_w] = image_resized

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(padded_image).permute(
        2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    return image_tensor, scale, (start_x, start_y)


def detect_id_card(image, models, device, expand_ratio=0.1):
    """Detects the ID card using YOLO, expands bounding box corners, crops, and corrects orientation."""

    image_tensor, scale, (pad_x, pad_y) = corner_preprocess_image(
        image, device)
    # Check for which model return the most boxes
    best_results = None
    max_boxes = 0

    for i, model in enumerate(models):
        results = model(image_tensor)

        # Extract unique boxes based on their coordinates
        unique_boxes = set(tuple(box.xyxy[0].tolist())
                           for box in results[0].boxes)
        num_unique_boxes = len(unique_boxes)

        if num_unique_boxes > max_boxes:
            max_boxes = num_unique_boxes
            best_results = results
    corners = []
    for result in best_results:
        for box in result.boxes.xyxy:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box)

            # Calculate center point
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Convert from padded coordinates back to original image coordinates
            orig_x = (center_x - pad_x) / scale
            orig_y = (center_y - pad_y) / scale

            corners.append([int(orig_x), int(orig_y)])
    if len(corners) <= 2:
        # Try all possible rotations (90, 180, 270 degrees)
        max_corners = corners
        best_image = image
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
        ]

        for rotation_code, angle in rotations:
            # Rotate image
            rotated_image = cv2.rotate(image, rotation_code)
            rotated_tensor, scale, (pad_x, pad_y) = corner_preprocess_image(
                rotated_image, device)
            rotated_results = model(rotated_tensor)

            # Detect corners in rotated image
            rotated_corners = []
            for result in rotated_results:
                for box in result.boxes.xyxy:
                    x_min, y_min, x_max, y_max = map(int, box)
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    orig_x = (center_x - pad_x) / scale
                    orig_y = (center_y - pad_y) / scale
                    rotated_corners.append([int(orig_x), int(orig_y)])

            if len(rotated_corners) > len(max_corners):

                max_corners = rotated_corners
                best_image = rotated_image

        # Use the rotation that gave us the most corners
        if len(max_corners) > len(corners):
            corners = max_corners
            image = best_image

        if len(corners) <= 2:
            return image

    if len(corners) == 3:

        new_corners = calculate_missed_coord_corner(corners)
        if len(new_corners) == 4:
            corners = new_corners
        else:
            return image

    if len(corners) >= 4:
        corners = np.array(corners, dtype="float32")

        if len(corners) > 4:
            # Find the best 4 corners that form the largest rectangle
            max_area = 0
            best_corners = None

            from itertools import combinations
            for four_corners in combinations(corners, 4):
                four_corners = np.array(four_corners)
                ordered = order_points(four_corners)
                # Calculate area using the ordered points
                (bl, br, tl, tr) = ordered
                width1 = np.linalg.norm(tr - tl)   # Top width
                width2 = np.linalg.norm(br - bl)   # Bottom width
                height1 = np.linalg.norm(tr - br)  # Right height
                height2 = np.linalg.norm(tl - bl)  # Left height

                # Calculate average area
                area = ((width1 + width2) / 2) * ((height1 + height2) / 2)

                # Keep track of largest area without angle verification
                if area > max_area:
                    max_area = area
                    best_corners = ordered

            if best_corners is not None:
                corners = best_corners
            else:
                return image
        else:
            corners = order_points(corners)

        center_x, center_y = np.mean(corners, axis=0)
        for i in range(4):
            direction = corners[i] - [center_x, center_y]
            corners[i] += direction * expand_ratio

        cropped_id = four_point_transform(image, corners)

        # Check QR code position and rotate if necessary
        final_id, rotation_angle = check_qr_position(cropped_id)
        if rotation_angle != 0:
            return final_id
        return cropped_id

    return image


def order_points(pts):
    """Orders points in order: (bottom-left, bottom-right, top-left, top-right)."""
    rect = np.zeros((4, 2), dtype='float32')

    # Find top-left and bottom-right using sum coordinates
    s = pts.sum(axis=1)
    temp_tl = pts[np.argmin(s)]  # Temporary top-left
    temp_br = pts[np.argmax(s)]  # Temporary bottom-right

    # Find top-right and bottom-left using difference of coordinates
    diff = np.diff(pts, axis=1)
    temp_tr = pts[np.argmin(diff)]  # Temporary top-right
    temp_bl = pts[np.argmax(diff)]  # Temporary bottom-left

    # Reorder to match [bottom_left, bottom_right, top_left, top_right]
    rect[0] = temp_bl  # bottom-left
    rect[1] = temp_br  # bottom-right
    rect[2] = temp_tl  # top-left
    rect[3] = temp_tr  # top-right

    return rect


def four_point_transform(image, pts):
    """Applies a perspective transform to get a top-down view of the ID."""
    rect = order_points(pts)
    # st.write("Ordered points:", rect)
    (bl, br, tl, tr) = rect  # Now in correct order

    # Compute vertical vector from bottom midpoint to top midpoint
    vertical_vector = (br + bl) / 2 - (tr + tl) / 2  # Vertical direction
    vertical_vector /= np.linalg.norm(vertical_vector)  # Normalize

    rect_extended = np.array([bl, br, tl, tr], dtype='float32')

    # Compute new dimensions
    widthA = np.linalg.norm(br - bl)  # Bottom width
    widthB = np.linalg.norm(tr - tl)  # Top width
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)  # Right height
    heightB = np.linalg.norm(tl - bl)  # Left height
    maxHeight = int(max(heightA, heightB))

 
    dst = np.array([
        [0, maxHeight - 1],             # bottom-left
        [maxWidth - 1, maxHeight - 1],  # bottom-right
        [0, 0],                         # top-left
        [maxWidth - 1, 0]               # top-right
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect_extended, dst)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return warped


def check_qr_position(image):
    """Check which quadrant contains the QR code and return required rotation degree and the rotated image
        Need to 4 corner image first"""
    height, width = image.shape[:2]
    mid_h, mid_w = height // 2, width // 2

    # Split image into quadrants
    top_left = image[0:mid_h, 0:mid_w]
    top_right = image[0:mid_h, mid_w:width]
    bottom_left = image[mid_h:height, 0:mid_w]
    bottom_right = image[mid_h:height, mid_w:width]

    # Initialize QR code reader
    qreader = QReader()

    # Check each quadrant for QR code
    quadrants = {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }

    qr_location = None
    for position, quad in quadrants.items():
        qr = qreader.detect_and_decode(quad)
        if qr is not None and len(qr) > 0:
            qr_location = position
            break

    # Determine rotation based on QR location
    if qr_location == 'top_right':
        return image, 0  # Correct orientation
    elif qr_location == 'bottom_left':
        return cv2.rotate(image, cv2.ROTATE_180), 180  # Rotate 180 degrees
    elif qr_location == 'bottom_right':  # Rotate left
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 90
    elif qr_location == 'top_left':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), -90  # Rotate right

    return image, 0


def calculate_missed_coord_corner(corners):
    """Calculates the missing fourth corner based on three corners.
    Returns corners in clockwise order starting from top-left."""
    if len(corners) != 3:
        return corners

    # Convert to numpy array
    corners = np.array(corners, dtype='float32')

    # Sort by y-coordinate
    sorted_y = corners[corners[:, 1].argsort()]

    # Get top points (smallest y values)
    top_points_mask = corners[:, 1] <= np.median(corners[:, 1])
    top_points = corners[top_points_mask]
    other_points = corners[~top_points_mask]

    if len(top_points) == 2:
        # We have two top points, need to calculate bottom point
        # Sort top points by x
        top_points = top_points[top_points[:, 0].argsort()]
        top_left, top_right = top_points

        # Vector from top-left to top-right
        top_vector = top_right - top_left

        # Other point is bottom-left or bottom-right
        bottom_point = other_points[0]
        if bottom_point[0] < np.mean([top_left[0], top_right[0]]):

            bottom_left = bottom_point
            bottom_right = bottom_left + top_vector
        else:

            bottom_right = bottom_point
            bottom_left = bottom_right - top_vector

    else:
        # We have one top point and two bottom points
        top_point = top_points[0]
        bottom_points = other_points[bottom_points[:, 0].argsort()]

        if len(bottom_points) < 2:
            return corners.tolist()  # Return original corners if we don't have enough points

        bottom_left, bottom_right = bottom_points

        # Determine if top point is top-left or top-right
        if top_point[0] < np.mean([bottom_left[0], bottom_right[0]]):

            top_left = top_point
            bottom_vector = bottom_right - bottom_left
            top_right = top_left + bottom_vector
        else:

            top_right = top_point
            bottom_vector = bottom_right - bottom_left
            top_left = top_right - bottom_vector

    # Return corners in clockwise order starting from top-left
    ordered_corners = np.array([
        top_left,     # top-left
        top_right,    # top-right
        bottom_right,  # bottom-right
        bottom_left   # bottom-left
    ])

    return ordered_corners.tolist()


def extract_yolo_results(results, image_shape):
    """ Extract boxes, scores, and class names from YOLO output. """
    boxes, scores, class_ids, class_names = [], [], [], []
    for result in results:

        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int)):

            x1, y1, x2, y2 = box

            boxes.append([x1 / image_shape[1], y1 / image_shape[0],
                          x2 / image_shape[1], y2 / image_shape[0]])
            scores.append(float(conf))
            class_ids.append(int(cls))
            class_names.append(result.names[cls])

    return boxes, scores, class_ids, class_names


def draw_yolo(results1, results2, image, vietocr_model):
    """ Run YOLO detections, fuse results, draw boxes, and extract text with VietOCR. """
    # Extract results from both YOLO models
    boxes1, scores1, labels1, names1 = extract_yolo_results(
        results1, image.shape)
    boxes2, scores2, labels2, names2 = extract_yolo_results(
        results2, image.shape)

    # Apply Weighted Boxes Fusion
    # Perform weighted boxes fusion on the results from both models
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [boxes1, boxes2],
        [scores1, scores2],
        [labels1, labels2],
        weights=[1, 1],
        iou_thr=0.5,
        skip_box_thr=0.3
    )

    # x1, y1, x2, y2 = max(0, x1-EXPEND), max(0,
    #                                         y1-EXPEND), x2+EXPEND, y2+EXPEND
    res = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = (int(box[0] * image.shape[1]), int(box[1] * image.shape[0]),
                          int(box[2] * image.shape[1]), int(box[3] * image.shape[0]))

        class_name = results1[0].names[label]
        if class_name == "qr":
            continue
        # Draw the rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{class_name} ({score:.2f})"

        # Extract and process text with VietOCR
        cropped_image = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(
            cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        text = vietocr_model.predict(pil_image)

        res.append(text)

    return image, res


def sharpen_image(image):
    """Sharpen the image using an unsharp mask."""
    gaussian_blurred = cv2.GaussianBlur(
        image, (0, 0), 3)  # Apply Gaussian blur
    sharpened = cv2.addWeighted(
        image, 1.5, gaussian_blurred, -0.5, 0)  # Add weighted mask
    return sharpened
