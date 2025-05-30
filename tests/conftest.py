"""
Test configuration and fixtures for Vietnamese ID Card OCR tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
import io

from src.config import Config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB image
    image = Image.new('RGB', (800, 600), color='white')
    return image


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    image = Image.new('RGB', (800, 600), color='white')
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format='PNG')
    return byte_buffer.getvalue()


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    return np.ones((600, 800, 3), dtype=np.uint8) * 255


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    mock_model = Mock()
    mock_model.predict.return_value = [Mock()]
    mock_model.predict.return_value[0].boxes = Mock()
    mock_model.predict.return_value[0].boxes.xyxy = np.array(
        [[100, 100, 200, 200]])
    mock_model.predict.return_value[0].boxes.conf = np.array([0.9])
    return mock_model


@pytest.fixture
def mock_vietocr_model():
    """Create a mock VietOCR model."""
    mock_model = Mock()
    mock_model.predict.return_value = "Sample Vietnamese text"
    return mock_model


@pytest.fixture
def sample_id_card_data():
    """Sample ID card extraction results."""
    return {
        'id_number': '123456789012',
        'full_name': 'Nguyễn Văn A',
        'date_of_birth': '01/01/1990',
        'gender': 'Nam',
        'nationality': 'Việt Nam',
        'place_of_origin': 'Hà Nội',
        'place_of_residence': '123 Đường ABC, Quận 1, TP.HCM',
        'issue_date': '01/01/2020',
        'expiry_date': '01/01/2030',
        'card_type': 'new'
    }


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini AI response."""
    return {
        'structured_info': {
            'id_number': '123456789012',
            'full_name': 'Nguyễn Văn A',
            'date_of_birth': '01/01/1990',
            'gender': 'Nam',
            'nationality': 'Việt Nam',
            'place_of_origin': 'Hà Nội',
            'place_of_residence': '123 Đường ABC, Quận 1, TP.HCM',
            'issue_date': '01/01/2020',
            'expiry_date': '01/01/2030'
        },
        'confidence_score': 0.95,
        'processing_notes': 'High quality extraction'
    }


class TestDatabase:
    """Test database utilities."""

    @staticmethod
    def create_test_db_connection():
        """Create a test database connection."""
        # This would connect to a test database
        pass

    @staticmethod
    def cleanup_test_data():
        """Clean up test data from database."""
        pass


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_IMAGES_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)
