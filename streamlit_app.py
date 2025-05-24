"""
Main entry point for Vietnamese ID Card OCR Streamlit application.
"""

from src.ui.streamlit_app import main
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


if __name__ == "__main__":
    main()
