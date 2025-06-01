"""
Centralized configuration settings for Vietnamese ID Card OCR application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    corner_detection_model: str = "models/corner_detection/weights/29_03_25-YOLOv11n-Corner-best_metrics.pt"
    text_detection_model: str = "models/text_detection/weights/best.pt"
    text_detection_model_v2: str = "models/text_detection_v2/weights/bestv2.pt"
    vietocr_config: str = "vgg_transformer"
    dictionary_path: str = "data/dictionary/dictionaries/hongocduc/words.txt"
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class APIConfig:
    """Configuration for API services."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class StreamlitConfig:
    """Configuration for Streamlit UI."""
    host: str = "0.0.0.0"
    port: int = 8501
    title: str = "Vietnamese ID Card OCR"
    icon: str = "🆔"
    layout: str = "wide"


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "vnid_card_ocr"
    enable_database: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for image processing."""
    max_image_size: tuple = (1920, 1080)
    confidence_threshold: float = 0.5
    enable_gemini: bool = True
    enable_image_enhancement: bool = True
    enable_qr_detection: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_header: str = "X-API-Key"
    rate_limit: str = "100/minute"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = field(default_factory=lambda: [
                                     '.jpg', '.jpeg', '.png', '.bmp'])


@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # API Keys
    google_ai_api_key: Optional[str] = os.getenv("GOOGLE_AI_API_KEY")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/app.log"

    # Component configs
    models: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories if they don't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/uploads", exist_ok=True)
        os.makedirs("data/outputs", exist_ok=True)

        # Override with environment variables
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API configuration
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", str(self.api.port)))

        # Streamlit configuration
        self.streamlit.host = os.getenv("STREAMLIT_HOST", self.streamlit.host)
        self.streamlit.port = int(
            os.getenv("STREAMLIT_PORT", str(self.streamlit.port)))

        # Database configuration
        self.database.mongodb_url = os.getenv(
            "MONGODB_URL", self.database.mongodb_url)
        self.database.database_name = os.getenv(
            "DATABASE_NAME", self.database.database_name)
        self.database.enable_database = os.getenv(
            "ENABLE_DATABASE", "true").lower() == "true"

        # Processing configuration
        confidence = os.getenv("CONFIDENCE_THRESHOLD")
        if confidence:
            self.processing.confidence_threshold = float(confidence)

        self.processing.enable_gemini = os.getenv(
            "ENABLE_GEMINI", "true").lower() == "true"

        # Model paths (allow override via env)
        model_dir = os.getenv("MODEL_DIR")
        if model_dir:
            self.models.corner_detection_model = f"{model_dir}/corner_detection_model/weight/29_03_25-YOLOv11n-Corner-best_metrics.pt"
            self.models.text_detection_model = f"{model_dir}/yolo_detect_text/best.pt"
            self.models.text_detection_model_v2 = f"{model_dir}/yolo_detect_text/bestv2.pt"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "models": {
                "corner_detection_model": self.models.corner_detection_model,
                "text_detection_model": self.models.text_detection_model,
                "text_detection_model_v2": self.models.text_detection_model_v2,
                "vietocr_config": self.models.vietocr_config,
                "dictionary_path": self.models.dictionary_path,
                "device": self.models.device,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
            },
            "database": {
                "mongodb_url": self.database.mongodb_url,
                "database_name": self.database.database_name,
                "enable_database": self.database.enable_database,
            },
            "processing": {
                "max_image_size": self.processing.max_image_size,
                "confidence_threshold": self.processing.confidence_threshold,
                "enable_gemini": self.processing.enable_gemini,
                "enable_image_enhancement": self.processing.enable_image_enhancement,
                "enable_qr_detection": self.processing.enable_qr_detection,
            }
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment."""
    global config
    config = AppConfig()
    return config
