"""
Configuration settings for Vietnamese ID Card OCR application.
This file maintains backward compatibility while the new config system is in config/settings.py
"""

import os
from pathlib import Path
from typing import Dict, Any

# Import the new configuration system
try:
    from ..config.settings import get_config, AppConfig
    NEW_CONFIG_AVAILABLE = True
except ImportError:
    NEW_CONFIG_AVAILABLE = False


class Config:
    """Configuration class for the application (legacy compatibility)."""

    def __init__(self):
        """Initialize configuration with environment variables."""
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.LOGS_DIR = self.BASE_DIR / "logs"

        # Model paths
        self.YOLO_CORNER_MODEL_PATH = self.BASE_DIR / "corner_detection_model" / \
            "weight" / "29_03_25-YOLOv11n-Corner-best_metrics.pt"
        self.YOLO_TEXT_MODEL_PATH = self.BASE_DIR / "yolo_detect_text" / "best.pt"
        self.YOLO_TEXT_V2_MODEL_PATH = self.BASE_DIR / "yolo_detect_text" / "bestv2.pt"
        self.DICTIONARY_PATH = self.BASE_DIR / "dictionary" / \
            "dictionaries" / "hongocduc" / "words.txt"

        # API Configuration
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8080"))
        self.METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

        # MongoDB Configuration
        self.MONGODB_URL = os.getenv(
            "MONGODB_URL", "mongodb://localhost:27017")
        self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "vnid_card_ocr")
        self.MONGODB_COLLECTION_RESULTS = os.getenv(
            "MONGODB_COLLECTION_RESULTS", "ocr_results")
        self.MONGODB_COLLECTION_SESSIONS = os.getenv(
            "MONGODB_COLLECTION_SESSIONS", "user_sessions")
        self.MONGODB_COLLECTION_METRICS = os.getenv(
            "MONGODB_COLLECTION_METRICS", "processing_metrics")

        # Processing Configuration
        self.DEFAULT_CONFIDENCE_THRESHOLD = 0.5
        self.DEFAULT_NMS_THRESHOLD = 0.3
        self.MAX_IMAGE_SIZE = (1920, 1080)
        self.SUPPORTED_IMAGE_FORMATS = [
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Device Configuration
        self.FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.MODEL_DIR, self.DATA_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_paths(self) -> Dict[str, Path]:
        """Get dictionary of model paths."""
        return {
            "yolo_corner": self.YOLO_CORNER_MODEL_PATH,
            "yolo_text": self.YOLO_TEXT_MODEL_PATH,
            "yolo_text_v2": self.YOLO_TEXT_V2_MODEL_PATH,
            "dictionary": self.DICTIONARY_PATH
        }

    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all required files exist."""
        model_paths = self.get_model_paths()
        validation_results = {}

        for name, path in model_paths.items():
            validation_results[name] = path.exists()

        return validation_results


# Development configuration
class DevelopmentConfig(Config):
    """Development-specific configuration."""

    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = "DEBUG"


# Production configuration
class ProductionConfig(Config):
    """Production-specific configuration."""

    def __init__(self):
        super().__init__()
        self.DEBUG = False
        self.LOG_LEVEL = "INFO"


# Test configuration
class TestConfig(Config):
    """Test-specific configuration."""

    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = "DEBUG"
        self.FORCE_CPU = True


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()
