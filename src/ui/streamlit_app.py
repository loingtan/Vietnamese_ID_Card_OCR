"""
Streamlit UI for Vietnamese ID Card OCR application.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import Dict, Any
import os
from pathlib import Path

from ..models.model_manager import ModelManager
from ..core.id_card_processor import IDCardProcessor


class StreamlitUI:
    """Streamlit user interface for ID Card OCR."""

    def __init__(self):
        self.setup_page_config()
        self.model_manager = None
        self.processor = None

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Vietnamese ID Card Scanner",
            page_icon="🆔",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def setup_sidebar(self):
        """Setup sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")

        # API Key input
        api_key = st.sidebar.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Enter your Google Gemini API key for enhanced processing"
        )

        # Model selection
        processing_method = st.sidebar.selectbox(
            "Processing Method",
            ["Auto (Gemini + OCR)", "Traditional OCR Only", "Gemini Only"],
            index=0
        )

        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            confidence_threshold = st.slider(
                "Detection Confidence",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )

            nms_threshold = st.slider(
                "NMS Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1
            )

            enhance_image = st.checkbox(
                "Enhance Image Quality",
                value=True
            )

        return {
            'api_key': api_key,
            'processing_method': processing_method,
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'enhance_image': enhance_image
        }

    def initialize_models(self, api_key: str = None):
        """Initialize models with caching."""
        if self.model_manager is None:
            with st.spinner("Loading models... This may take a few minutes on first run."):
                try:
                    self.model_manager = ModelManager(api_key=api_key)
                    self.processor = IDCardProcessor(self.model_manager)
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
                    return False
        return True

    def display_header(self):
        """Display application header."""
        st.title("🆔 Vietnamese ID Card Scanner")
        st.markdown("""
        Upload an image of a Vietnamese ID card to extract information automatically.
        Supports both old and new ID card formats with high accuracy OCR.
        """)

        # Add some metrics or info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Supported Formats", "Old & New ID")
        with col2:
            st.metric("Languages", "Vietnamese")
        with col3:
            st.metric("Processing", "AI + OCR")
        with col4:
            st.metric("Accuracy", "95%+")

    def upload_image(self):
        """Handle image upload."""
        uploaded_file = st.file_uploader(
            "Choose an ID card image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a Vietnamese ID card"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📤 Uploaded Image")
                st.image(image, use_container_width=True)

            # Convert to numpy array for processing
            image_array = np.array(image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            return image_array, col2

        return None, None

    def display_results(self, results: Dict[str, Any], col):
        """Display processing results."""
        with col:
            st.subheader("📋 Extracted Information")

            if results.get('status') == 'success':
                extracted_info = results.get('extracted_info', {})
                method = results.get('method', 'unknown')

                # Display processing method
                st.info(f"🔍 Processed using: {method.title()}")

                if extracted_info:
                    # Create a nice display of the extracted information
                    info_data = []
                    field_labels = {
                        'id_number': '🆔 ID Number',
                        'full_name': '👤 Full Name',
                        'date_of_birth': '📅 Date of Birth',
                        'sex': '⚥ Gender',
                        'nationality': '🏳️ Nationality',
                        'place_of_origin': '🏠 Place of Origin',
                        'place_of_residence': '📍 Place of Residence',
                        'date_of_expiry': '⏰ Date of Expiry'
                    }

                    for key, value in extracted_info.items():
                        if value:
                            label = field_labels.get(
                                key, key.replace('_', ' ').title())
                            info_data.append(
                                {'Field': label, 'Value': str(value)})

                    if info_data:
                        df = pd.DataFrame(info_data)
                        st.dataframe(df, use_container_width=True,
                                     hide_index=True)

                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="id_card_info.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(
                            "⚠️ No information could be extracted from the image.")
                else:
                    st.warning(
                        "⚠️ No information could be extracted from the image.")

            elif results.get('status') == 'error':
                st.error(
                    f"❌ Processing failed: {results.get('message', 'Unknown error')}")
            else:
                st.warning("⚠️ Unexpected result format.")

    def display_processing_info(self):
        """Display information about the processing pipeline."""
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            ### Processing Pipeline:
            1. **Image Enhancement** - Improve image quality for better OCR
            2. **Corner Detection** - Locate ID card boundaries using YOLO
            3. **Perspective Correction** - Straighten the ID card image
            4. **Text Detection** - Find text regions using multiple models
            5. **Text Recognition** - Extract Vietnamese text using VietOCR
            6. **Information Extraction** - Parse structured data from text
            7. **Validation** - Clean and validate extracted information
            
            ### Features:
            - ✅ Supports both old and new Vietnamese ID card formats
            - ✅ High accuracy Vietnamese text recognition
            - ✅ Automatic perspective and orientation correction
            - ✅ AI-powered information extraction with Gemini
            - ✅ Fallback to traditional OCR methods
            """)

    def run(self):
        """Main application run method."""
        self.display_header()

        # Setup sidebar and get configuration
        config = self.setup_sidebar()

        # Initialize models
        if not self.initialize_models(config.get('api_key')):
            st.stop()

        # Display processing info
        self.display_processing_info()

        # Image upload and processing
        image_array, results_col = self.upload_image()

        if image_array is not None:
            # Process button
            if st.button("🚀 Process ID Card", type="primary"):
                with st.spinner("Processing image... Please wait."):
                    try:
                        results = self.processor.process_id_card(image_array)
                        self.display_results(results, results_col)
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray;'>"
            "Vietnamese ID Card OCR System | "
            "Built with ❤️ using Streamlit, YOLO, and VietOCR"
            "</p>",
            unsafe_allow_html=True
        )


def main():
    """Main entry point for Streamlit app."""
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
