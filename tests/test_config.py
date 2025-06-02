"""
Tests for configuration management.
"""

from config import Config, get_config, DevelopmentConfig, ProductionConfig, TestConfig as TestConfigClass
import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfiguration:
    """Test configuration class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = Config()
        assert config.API_HOST == "0.0.0.0"
        assert config.API_PORT == 8080
        assert config.LOG_LEVEL == "INFO"
        assert config.DEFAULT_CONFIDENCE_THRESHOLD == 0.5

    def test_environment_variable_override(self):
        """Test configuration override from environment variables."""
        with patch.dict(os.environ, {
            'API_PORT': '9000',
            'LOG_LEVEL': 'DEBUG',
            'FORCE_CPU': 'true'
        }):
            config = Config()
            assert config.API_PORT == 9000
            assert config.LOG_LEVEL == "DEBUG"
            assert config.FORCE_CPU is True

    def test_model_paths_exist(self):
        """Test that model paths are properly configured."""
        config = Config()
        model_paths = config.get_model_paths()

        assert "yolo_corner" in model_paths
        assert "yolo_text" in model_paths
        assert "yolo_text_v2" in model_paths
        assert "dictionary" in model_paths

        # Check that paths are Path objects
        for path in model_paths.values():
            assert isinstance(path, Path)

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            # Mock the BASE_DIR to use temp directory
            config.BASE_DIR = Path(temp_dir)
            config.MODEL_DIR = config.BASE_DIR / "models"
            config.DATA_DIR = config.BASE_DIR / "data"
            config.LOGS_DIR = config.BASE_DIR / "logs"

            config.ensure_directories()

            assert config.MODEL_DIR.exists()
            assert config.DATA_DIR.exists()
            assert config.LOGS_DIR.exists()

    def test_validate_setup(self):
        """Test setup validation."""
        config = Config()
        validation_results = config.validate_setup()

        assert isinstance(validation_results, dict)
        assert "yolo_corner" in validation_results
        assert "yolo_text" in validation_results
        assert "yolo_text_v2" in validation_results
        assert "dictionary" in validation_results

        # Results should be boolean
        for result in validation_results.values():
            assert isinstance(result, bool)


class TestConfigEnvironments:
    """Test different configuration environments."""

    def test_development_config(self):
        """Test development configuration."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            config = get_config()
            assert config.__class__.__name__ == "DevelopmentConfig"
            assert config.DEBUG is True
            assert config.LOG_LEVEL == "DEBUG"

    def test_production_config(self):
        """Test production configuration."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config = get_config()
            assert config.__class__.__name__ == "ProductionConfig"
            assert config.DEBUG is False
            assert config.LOG_LEVEL == "INFO"

    def test_test_config(self):
        """Test test configuration."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            config = get_config()
            assert config.__class__.__name__ == "TestConfig"
            assert config.DEBUG is True
            assert config.LOG_LEVEL == "DEBUG"
            assert config.FORCE_CPU is True

    def test_default_environment(self):
        """Test default environment fallback."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            assert config.__class__.__name__ == "DevelopmentConfig"


class TestConfigValidation:
    """Test configuration validation methods."""

    def test_supported_image_formats(self):
        """Test supported image formats configuration."""
        config = Config()
        formats = config.SUPPORTED_IMAGE_FORMATS

        assert ".jpg" in formats
        assert ".jpeg" in formats
        assert ".png" in formats
        assert ".bmp" in formats
        assert ".tiff" in formats

    def test_max_image_size(self):
        """Test maximum image size configuration."""
        config = Config()
        max_size = config.MAX_IMAGE_SIZE

        assert isinstance(max_size, tuple)
        assert len(max_size) == 2
        assert max_size[0] == 1920
        assert max_size[1] == 1080

    def test_mongodb_configuration(self):
        """Test MongoDB configuration."""
        config = Config()

        assert config.MONGODB_URL == "mongodb://localhost:27017"
        assert config.MONGODB_DATABASE == "vnid_card_ocr"
        assert config.MONGODB_COLLECTION_RESULTS == "ocr_results"
        assert config.MONGODB_COLLECTION_SESSIONS == "user_sessions"
        assert config.MONGODB_COLLECTION_METRICS == "processing_metrics"

    def test_api_configuration(self):
        """Test API configuration."""
        config = Config()

        assert config.API_HOST == "0.0.0.0"
        assert config.API_PORT == 8080
        assert config.METRICS_PORT == 8000
        assert config.GEMINI_API_KEY == ""  # Default empty
