"""
Streamlit UI for Vietnamese ID Card OCR application.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import Dict, Any, List, Tuple
import os
from pathlib import Path
import argparse
import uuid
import time
from datetime import datetime
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.model_manager import ModelManager
from ..core.id_card_processor import IDCardProcessor
from src.database import MongoDBClient, OCRResult, UserSession
from ..config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StreamlitUI:
    """Streamlit user interface for ID Card OCR."""

    def __init__(self, port: int = 8501):
        self.port = port
        self.config = get_config()
        self.setup_page_config()
        
        # Initialize model state in session state
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = None
        if 'processor' not in st.session_state:
            st.session_state.processor = None
            
        self.model_manager = st.session_state.model_manager
        self.processor = st.session_state.processor
        
        self.db_client = MongoDBClient()
        try:
            self.db_client.connect()
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            st.error(f"Failed to connect to MongoDB: {e}")
        
        # Initialize session
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            try:
                session = UserSession(
                    session_id=st.session_state.session_id,
                    created_at=datetime.utcnow()
                )
                self.db_client.save_session(session)
                logger.info(f"Created new session: {st.session_state.session_id}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                st.error(f"Failed to create session: {e}")

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

        # Add navigation with styled buttons
        st.sidebar.markdown("### 📱 Navigation")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            scan_button = st.button(
                "🆔 Scan ID Card",
                use_container_width=True,
                help="Upload and process ID card images"
            )
        
        with col2:
            history_button = st.button(
                "📚 View History",
                use_container_width=True,
                help="View processing history"
            )
        
        # Set active page based on button clicks
        if 'active_page' not in st.session_state:
            st.session_state.active_page = "Scan ID Card"
            
        if scan_button:
            st.session_state.active_page = "Scan ID Card"
        elif history_button:
            st.session_state.active_page = "View History"
            
        # Add visual feedback for active page
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Current Page:** {st.session_state.active_page}")

        # API Key input
        st.sidebar.markdown("### 🔑 API Settings")
        api_key = st.sidebar.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Enter your Google Gemini API key for enhanced processing"
        )

        # Model selection
        st.sidebar.markdown("### 🛠️ Processing Settings")
        processing_method = st.sidebar.selectbox(
            "Processing Method",
            ["Auto (Gemini + OCR)", "Traditional OCR Only", "Gemini Only"],
            index=0
        )

        # Advanced settings
        with st.sidebar.expander("⚡ Advanced Settings"):
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
            'page': st.session_state.active_page,
            'api_key': api_key,
            'processing_method': processing_method,
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'enhance_image': enhance_image
        }

    def initialize_models(self, api_key: str = None):
        """Initialize models with caching."""
        if st.session_state.model_manager is None:
            with st.spinner("Loading models... This may take a few minutes on first run."):
                try:
                    st.session_state.model_manager = ModelManager(api_key=api_key)
                    st.session_state.processor = IDCardProcessor(st.session_state.model_manager)
                    self.model_manager = st.session_state.model_manager
                    self.processor = st.session_state.processor
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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Supported Formats", "Old & New ID")
        with col2:
            st.metric("Languages", "Vietnamese")

    def process_single_image(self, image: np.ndarray, idx: int, total_images: int) -> Dict[str, Any]:
        """Process a single image with progress tracking."""
        try:
            # Process the image
            result = self.processor.process_id_card(image)
            if result and 'extracted_info' in result:
                return {
                    'status': 'success',
                    'extracted_info': result['extracted_info'],
                    'message': 'Successfully processed',
                    'index': idx
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to extract information',
                    'index': idx
                }
        except Exception as e:
            logger.error(f"Error processing image {idx + 1}: {e}")
            return {
                'status': 'error',
                'message': f"Error: {str(e)}",
                'index': idx
            }

    def process_batch_images(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a batch of images using multithreading."""
        results = []
        total_images = len(images)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_count = 0
        
        # Automatically determine if we should use multithreading
        # Use multithreading if we have more than 1 image
        max_workers = min(os.cpu_count() or 4, total_images) if total_images > 1 else 1
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(self.process_single_image, image, idx, total_images): idx 
                    for idx, image in enumerate(images)
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_image):
                    result = future.result()
                    results.append(result)
                    processed_count += 1
                    
                    # Update progress
                    progress = processed_count / total_images
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {processed_count} of {total_images}")
                    
                    # Log progress
                    logger.info(f"Completed processing image {processed_count}/{total_images}")
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            st.error(f"Error in batch processing: {str(e)}")
        
        finally:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x.get('index', 0))
        return results

    def upload_images(self) -> Tuple[List[np.ndarray], Any]:
        """Handle multiple image uploads."""
        uploaded_files = st.file_uploader(
            "Choose ID card images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload one or more images of Vietnamese ID cards"
        )

        if uploaded_files:
            # Display uploaded images
            images = []
            cols = st.columns(min(3, len(uploaded_files)))
            
            for idx, uploaded_file in enumerate(uploaded_files):
                col = cols[idx % 3]
                with col:
                    st.subheader(f"Image {idx + 1}")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                    # Convert to numpy array for processing
                    image_array = np.array(image)
                    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        # Convert RGB to BGR for OpenCV
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        images.append(image_array)

            return images, st.container()

        return None, None

    def display_history(self):
        """Display history of processed ID cards."""
        st.title("📚 Processing History")
        
        try:
            # Get results from MongoDB
            results = self.db_client.get_ocr_results_by_session(st.session_state.session_id)
            logger.info(f"Retrieved {len(results)} results for session {st.session_state.session_id}")
            
            if not results:
                st.info("No processing history found.")
                return
                
            # Convert results to DataFrame
            history_data = []
            for result in results:
                info = result.get('extracted_info', {})
                history_data.append({
                    'Timestamp': result.get('timestamp', ''),
                    'ID Number': info.get('id_number', ''),
                    'Full Name': info.get('full_name', ''),
                    'Date of Birth': info.get('date_of_birth', ''),
                    'Gender': info.get('sex', ''),
                    'Nationality': info.get('nationality', ''),
                    'Place of Origin': info.get('place_of_origin', ''),
                    'Place of Residence': info.get('place_of_residence', ''),
                    'Date of Expiry': info.get('date_of_expiry', '')
                })
                
            if history_data:
                df = pd.DataFrame(history_data)
                df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download History as CSV",
                    data=csv,
                    file_name="id_card_history.csv",
                    mime="text/csv"
                )
            else:
                st.info("No processing history found.")
        except Exception as e:
            logger.error(f"Error displaying history: {e}")
            st.error(f"Error displaying history: {e}")

    def check_duplicate_id(self, id_number: str) -> Dict[str, Any]:
        """Check if ID number already exists in database."""
        try:
            results = self.db_client.search_by_id_number(id_number)
            if results:
                # Get the most recent result
                latest_result = results[0]
                return {
                    'is_duplicate': True,
                    'previous_result': latest_result,
                    'total_occurrences': len(results)
                }
            return {'is_duplicate': False}
        except Exception as e:
            logger.error(f"Error checking duplicate ID: {e}")
            return {'is_duplicate': False, 'error': str(e)}

    def display_duplicate_warning(self, duplicate_info: Dict[str, Any]):
        """Display warning for duplicate ID card."""
        if duplicate_info.get('is_duplicate'):
            st.warning("⚠️ This ID card has been processed before!")
            
            # Show previous processing details
            with st.expander("View Previous Processing Details"):
                prev_result = duplicate_info['previous_result']
                info = prev_result.get('extracted_info', {})
                
                # Display previous processing info
                st.write(f"**Last Processed:** {prev_result.get('timestamp', 'Unknown')}")
                st.write(f"**Total Occurrences:** {duplicate_info['total_occurrences']}")
                
                # Display extracted information
                st.write("**Extracted Information:**")
                for key, value in info.items():
                    if value:
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")

    def display_batch_results(self, results: List[Dict[str, Any]], container):
        """Display results for a batch of processed images."""
        with container:
            st.subheader("📋 Batch Processing Results")
            
            # Create tabs for each image result
            tabs = st.tabs([f"Image {i+1}" for i in range(len(results))])
            
            for idx, (tab, result) in enumerate(zip(tabs, results)):
                with tab:
                    if result.get('status') == 'success':
                        extracted_info = result.get('extracted_info', {})
                        
                        # Check for duplicate ID
                        id_number = extracted_info.get('id_number')
                        if id_number:
                            duplicate_info = self.check_duplicate_id(id_number)
                            self.display_duplicate_warning(duplicate_info)
                        
                        # Display extracted information
                        if extracted_info:
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
                                    label = field_labels.get(key, key.replace('_', ' ').title())
                                    info_data.append({'Field': label, 'Value': str(value)})

                            if info_data:
                                df = pd.DataFrame(info_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                
                                try:
                                    # Save to MongoDB
                                    ocr_result = OCRResult(
                                        session_id=st.session_state.session_id,
                                        image_filename=f"uploaded_image_{idx+1}.jpg",
                                        extracted_info=extracted_info,
                                        processing_time=0.0,
                                        confidence_scores={},
                                        detected_text_regions=[],
                                        success=True
                                    )
                                    result_id = self.db_client.save_ocr_result(ocr_result)
                                    logger.info(f"Saved OCR result with ID: {result_id}")
                                    st.success("✅ Results saved to database!")
                                except Exception as e:
                                    logger.error(f"Failed to save to MongoDB: {e}")
                                    st.error(f"Failed to save to database: {e}")
                                
                                # Download button for individual result
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download Image {idx+1} Results",
                                    data=csv,
                                    file_name=f"id_card_info_{idx+1}.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.warning("⚠️ No information could be extracted from the image.")
                    else:
                        st.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            
            # Add batch download option
            if any(r.get('status') == 'success' for r in results):
                all_data = []
                for idx, result in enumerate(results):
                    if result.get('status') == 'success':
                        info = result.get('extracted_info', {})
                        info['Image_Number'] = idx + 1
                        all_data.append(info)
                
                if all_data:
                    batch_df = pd.DataFrame(all_data)
                    batch_csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Results",
                        data=batch_csv,
                        file_name="batch_id_card_results.csv",
                        mime="text/csv"
                    )

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
            - ✅ Automatuse_container_width=Trueic perspective and orientation correction
            - ✅ AI-powered information extraction with Gemini
            - ✅ Fallback to traditional OCR methods
            """)

    def run(self):
        """Main application run method."""
        # Setup sidebar and get configuration
        config = self.setup_sidebar()

        # Handle navigation
        if config['page'] == "View History":
            self.display_history()
            return

        self.display_header()

        # Initialize models only if not already initialized
        if not self.initialize_models(config.get('api_key')):
            st.stop()

        # Display processing info
        self.display_processing_info()

        # Image upload and processing
        images, results_container = self.upload_images()

        if images:
            # Process button
            if st.button("🚀 Process ID Cards", type="primary"):
                with st.spinner("Processing images... Please wait."):
                    try:
                        # Process the batch
                        results = self.process_batch_images(images)
                        
                        # Display results
                        self.display_batch_results(results, results_container)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing images: {str(e)}")
        else:
            # Display a message when no images are uploaded
            st.info("👆 Please upload one or more ID card images to begin processing.")

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
    """Main entry point for the Streamlit application."""
    parser = argparse.ArgumentParser(description='Vietnamese ID Card OCR Streamlit App')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the Streamlit app on')
    args = parser.parse_args()
    
    ui = StreamlitUI(port=args.port)
    ui.run()


if __name__ == "__main__":
    main()

__all__ = ['main']
