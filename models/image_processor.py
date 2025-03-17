import cv2
import numpy as np
import streamlit as st
from utils.text_extractor import extract_field_info


def process_image(image, paddle_model, vietocr_model):
    """Process the image using PaddleOCR and VietOCR"""
    if paddle_model is None or vietocr_model is None:
        st.error("Models not loaded correctly")
        return None

    # Step 1: Use PaddleOCR for text detection
    result = paddle_model.ocr(image, cls=True)

    if not result or len(result) == 0 or result[0] is None:
        st.warning("No text detected in the image")
        return None

    # Extract bounding boxes
    boxes = [line[0] for line in result[0]]

    # Step 2: Crop detected regions and use VietOCR for text recognition
    extracted_texts = []

    # Create a copy of the image for visualization
    vis_image = image.copy()

    for idx, box in enumerate(boxes):
        # Convert box to proper format
        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the box on the visualization image
        cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
        cv2.putText(vis_image, str(idx), (points[0][0][0], points[0][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Create mask and crop region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Get bounding rect
        x, y, w, h = cv2.boundingRect(points)
        cropped = masked[y:y + h, x:x + w]

        # Skip if cropped image is too small
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        try:
            text = vietocr_model.predict(cropped)
            if text and len(text.strip()) > 0:
                extracted_texts.append(text)
        except Exception as e:
            st.warning(f"Error recognizing text in region {idx}: {e}")

    structured_info = extract_field_info(extracted_texts)

    return {
        "visualization": vis_image,
        "texts": extracted_texts,
        "structured_info": structured_info
    }
