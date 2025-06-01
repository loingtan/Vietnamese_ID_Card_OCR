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
        self.YOLO_CORNER_MODEL_PATH = self.MODEL_DIR / "yolo_corner_detect" / \
            "weights" / "29_03_25-YOLOv11n-Corner-best_metrics.pt"
        self.YOLO_TEXT_MODEL_PATH = self.MODEL_DIR / "yolo_text_detect"/ "weights" / "best.pt"
        self.YOLO_TEXT_V2_MODEL_PATH = self.MODEL_DIR / "yolo_text_detect_v2"/ "weights" / "bestv2.pt"
        self.DICTIONARY_PATH = self.DATA_DIR / "dictionary" / \
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
        
        # DVC
        self.YOLO_CORNER_MODEL_DATASET =  self.MODEL_DIR / "yolo_corner_detection" / "datasets"
        self.YOLO_TEXT_MODEL_DATASET = self.MODEL_DIR / "yolo_text_detection" / "datasets"
        self.YOLO_TEXT_MODEL_V2_DATASET = self.MODEL_DIR / "yolo_text_detection_v2" / "datasets"

        # === MLflow Integration ===
        self.MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
        self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.MLFLOW_MODEL_ARTIFACTS = {
            "yolo_text_detect": {
                "version": "1",
                "run_id": os.getenv("MLFLOW_RUN_YOLO_TEXT", ""),
                "artifact_path": ""
            },
            "yolo_text_detect_v2": {
                "version": "1",
                "run_id": os.getenv("MLFLOW_RUN_YOLO_TEXT_V2", ""),
                "artifact_path": ""
            },
            "yolo_corner_detect": {
                "version": "1",
                "run_id": os.getenv("MLFLOW_RUN_YOLO_CORNER", ""),
                "artifact_path": ""
            }
        }

    def ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.MODEL_DIR, self.DATA_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_paths(self) -> Dict[str, Path]:
        """Get dictionary of model paths."""
        return {
            "yolo_corner_detect": self.YOLO_CORNER_MODEL_PATH,
            "yolo_text_detect": self.YOLO_TEXT_MODEL_PATH,
            "yolo_text_detect_v2": self.YOLO_TEXT_V2_MODEL_PATH,
            "dictionary": self.DICTIONARY_PATH
        }
        
    @classmethod
    def get_mlflow_model_config(self) -> Dict[str, Dict[str, str]]:
        """Return MLflow model mapping (run_id + artifact path)."""
        return self.MLFLOW_MODEL_ARTIFACTS

    @classmethod
    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all required files exist."""
        model_paths = self.get_model_paths()
        validation_results = {}

        for name, path in model_paths.items():
            validation_results[name] = path.exists()

        return validation_results


class DevelopmentConfig(Config):
    """Development-specific configuration."""

    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production-specific configuration."""

    def __init__(self):
        super().__init__()
        self.DEBUG = False
        self.LOG_LEVEL = "INFO"


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
