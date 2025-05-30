# This is a backup of the original monolithic app.py file
# The original file has been refactored into a modular structure
# If you need to reference the original code, it's preserved here

# Original app.py was 1497 lines long and contained:
# - Model loading and management
# - Image processing utilities
# - Text processing and correction
# - OCR pipeline
# - Streamlit UI
# - FastAPI components
# - Configuration management

# The code has been refactored into:
# - src/models/model_manager.py - Model management
# - src/utils/image_processing.py - Image processing utilities
# - src/utils/text_processing.py - Text processing utilities
# - src/core/id_card_processor.py - Main OCR pipeline
# - src/ui/streamlit_app.py - Streamlit interface
# - src/api/fastapi_app.py - FastAPI application
# - src/config.py - Configuration management

# To access the original code, use: git show HEAD:app.py
