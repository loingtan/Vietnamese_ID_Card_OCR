"""
Configuration settings for Vietnamese ID Card OCR application.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the application."""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"

    # Model paths (for local fallback)
    YOLO_CORNER_MODEL_PATH = BASE_DIR / "models" / "corner_detection" / "weights" / "29_03_25-YOLOv11n-Corner-best_metrics.pt"
    YOLO_TEXT_MODEL_PATH = BASE_DIR / "models" / "text_detection" / "weights" / "best.pt"
    YOLO_TEXT_V2_MODEL_PATH = BASE_DIR / "models" / "text_detection" / "weights" / "bestv2.pt"
    DICTIONARY_PATH = BASE_DIR / "dictionary" / "dictionaries" / "hongocduc" / "words.txt"

    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

    # MongoDB Configuration
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "vnid_card_ocr")
    MONGODB_COLLECTION_RESULTS = os.getenv("MONGODB_COLLECTION_RESULTS", "ocr_results")
    MONGODB_COLLECTION_SESSIONS = os.getenv("MONGODB_COLLECTION_SESSIONS", "user_sessions")
    MONGODB_COLLECTION_METRICS = os.getenv("MONGODB_COLLECTION_METRICS", "processing_metrics")

    # Processing Configuration
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_NMS_THRESHOLD = 0.3
    MAX_IMAGE_SIZE = (1920, 1080)
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Device Configuration
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # === MLflow Integration ===
    MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Manual-Upload")

    # Map model keys to MLflow run ID + artifact path
    MLFLOW_MODEL_ARTIFACTS = {
        "yolo_text": {
            "run_id": os.getenv("MLFLOW_RUN_YOLO_TEXT", ""),
            "artifact_path": "models/best.pt"
        },
        "yolo_text_v2": {
            "run_id": os.getenv("MLFLOW_RUN_YOLO_TEXT_V2", ""),
            "artifact_path": "models/bestv2.pt"
        },
        "yolo_corner": {
            "run_id": os.getenv("MLFLOW_RUN_YOLO_CORNER", ""),
            "artifact_path": "models/29_03_25-YOLOv11n-Corner-best_metrics.pt"
        }
    }

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.MODEL_DIR, cls.DATA_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_paths(cls) -> Dict[str, Path]:
        """Get dictionary of model paths."""
        return {
            "yolo_corner": cls.YOLO_CORNER_MODEL_PATH,
            "yolo_text": cls.YOLO_TEXT_MODEL_PATH,
            "yolo_text_v2": cls.YOLO_TEXT_V2_MODEL_PATH,
            "dictionary": cls.DICTIONARY_PATH
        }

    @classmethod
    def get_mlflow_model_config(cls) -> Dict[str, Dict[str, str]]:
        """Return MLflow model mapping (run_id + artifact path)."""
        return cls.MLFLOW_MODEL_ARTIFACTS

    @classmethod
    def validate_setup(cls) -> Dict[str, bool]:
        """Validate that all required files exist."""
        model_paths = cls.get_model_paths()
        validation_results = {}

        for name, path in model_paths.items():
            validation_results[name] = path.exists()

        return validation_results


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"


class TestConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    FORCE_CPU = True


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()
