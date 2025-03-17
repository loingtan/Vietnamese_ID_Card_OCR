import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Import custom modules
from models.model_loader import load_paddle_model, load_vietocr_model
from models.image_processor import process_image
from utils.image_utils import preprocess_image

# Set up page configuration
st.set_page_config(
    page_title="Vietnamese ID Card Scanner",
    page_icon="🆔",
    layout="wide"
)

# Application title and description
st.title("Vietnamese ID Card Scanner")
st.write("Upload an image of a Vietnamese ID card for OCR processing")

# Sidebar for model path input
with st.sidebar:
    st.header("Model Configuration")
    model_path = st.text_input(
        "Path to PaddleOCR pretrained model",
        value="inference_model"
    )
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

# Load models
paddle_model = load_paddle_model(model_path)
vietocr_model = load_vietocr_model()

# Main application logic
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for display
    col1, col2 = st.columns(2)

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Process button
    if st.button("Process ID Card"):
        with st.spinner('Processing image...'):
            # Process the image
            result = process_image(image, paddle_model, vietocr_model)

            if result:
                # Display visualized image with bounding boxes
                with col2:
                    st.subheader("Detected Regions")
                    st.image(cv2.cvtColor(
                        result["visualization"], cv2.COLOR_BGR2RGB), use_column_width=True)

                # Display extracted information
                st.subheader("Extracted Information")

                # Display structured information in a table
                for field, value in result["structured_info"].items():
                    if value:
                        st.info(f"**{field}**: {value}")
                    else:
                        st.warning(f"**{field}**: Not detected")

                # Show all detected text
                with st.expander("All Detected Text"):
                    for idx, text in enumerate(result["texts"]):
                        st.write(f"{idx + 1}. {text}")

                # Option to download results as CSV
                df = pd.DataFrame([result["structured_info"]])
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "id_card_results.csv",
                    "text/csv",
                    key='download-csv'
                )
else:
    st.info("Please upload an image to begin processing")

# Add instructions at the bottom
with st.expander("Instructions"):
    st.markdown("""
    ### How to use this application:
    1. Set the path to your pretrained PaddleOCR model in the sidebar
    2. Upload an image of a Vietnamese ID card
    3. Click "Process ID Card" to start OCR
    4. Review the extracted information
    5. Download results as CSV if needed

    ### Note:
    - For best results, ensure the ID card image is clear and well-lit
    - The application works with both old and new Vietnamese ID card formats
    - Adjust the confidence threshold in the sidebar if needed
    """)
