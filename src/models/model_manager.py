"""
Model loading and management utilities for Vietnamese ID Card OCR.
"""

import torch
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import pipeline
from google import genai
import streamlit as st
from pathlib import Path
import os

# Import the new configuration system
try:
    from ...config.settings import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    # Fallback for legacy imports
    CONFIG_AVAILABLE = False


class ModelManager:
    """Manages all models used in the Vietnamese ID Card OCR system."""

    def __init__(self, api_key: str = None, config=None):
        if CONFIG_AVAILABLE and config is None:
            self.config = get_config()
        else:
            self.config = config

        # Set device based on config or auto-detection
        if hasattr(self.config, 'models') and self.config.models.device != "auto":
            self.device = self.config.models.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = {}
        self.api_key = api_key or (self.config.google_ai_api_key if hasattr(
            self.config, 'google_ai_api_key') else None)
        self._load_all_models()

    def _load_all_models(self):
        """Load all required models."""
        self.models['vietocr'] = self._load_vietocr_model()
        self.models['yolo_text_detect'] = self._load_yolo_text_detection_model()
        self.models['yolo_text_detect_v2'] = self._load_yolo_text_detection_model_v2()
        self.models['yolo_corner_detect'] = self._load_yolo_corner_detection_model()
        self.models['text_corrector'] = self._load_text_correction_model()
        if self.api_key:
            self.models['gemini_client'] = self._load_gemini_client()

    @st.cache_resource
    def _load_vietocr_model(_self):
        """Load the VietOCR model for text recognition."""
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = True
            config['device'] = _self.device
            predictor = Predictor(config)
            return predictor
        except Exception as e:
            st.error(f"Error loading VietOCR model: {e}")
            return None

    @st.cache_resource
    def _load_yolo_text_detection_model(_self):
        """Load YOLO model for text detection."""
        try:
            model_path = Path("yolo_detect_text/best.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(_self.device)
            return model
        except Exception as e:
            st.error(f"Error loading YOLO text detection model: {e}")
            return None

    @st.cache_resource
    def _load_yolo_text_detection_model_v2(_self):
        """Load YOLO v2 model for text detection."""
        try:
            model_path = Path("yolo_detect_text/bestv2.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(_self.device)
            return model
        except Exception as e:
            st.error(f"Error loading YOLO text detection model v2: {e}")
            return None

    @st.cache_resource
    def _load_yolo_corner_detection_model(_self):
        """Load YOLO model for ID card corner detection."""
        try:
            model_path = Path(
                "corner_detection_model/weight/29_03_25-YOLOv11n-Corner-best_metrics.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(_self.device)
            return [model]  # Return as list for compatibility
        except Exception as e:
            st.error(f"Error loading YOLO corner detection model: {e}")
            return None

    @st.cache_resource
    def _load_text_correction_model(_self):
        """Load Vietnamese text correction model."""
        try:
            corrector = pipeline(
                "text2text-generation",
                model="bmd1905/vietnamese-correction-v2",
                device=0 if _self.device == "cuda" else -1
            )
            return corrector
        except Exception as e:
            st.error(f"Error loading text correction model: {e}")
            return None

    def _load_gemini_client(self):
        """Load Gemini client for AI processing."""
        try:
            if not self.api_key:
                return None
            client = genai.Client(api_key=self.api_key)
            return client
        except Exception as e:
            st.error(f"Error loading Gemini client: {e}")
            return None

    def get_model(self, model_name: str):
        """Get a specific model by name."""
        return self.models.get(model_name)

    def get_device(self):
        """Get the current device (cuda/cpu)."""
        return self.device

    def reload_model(self, model_name: str):
        """Reload a specific model."""
        if model_name == 'vietocr':
            self.models['vietocr'] = self._load_vietocr_model()
        elif model_name == 'yolo_text_detect':
            self.models['yolo_text_detect'] = self._load_yolo_text_detection_model()
        elif model_name == 'yolo_text_detect_v2':
            self.models['yolo_text_detect_v2'] = self._load_yolo_text_detection_model_v2()
        elif model_name == 'yolo_corner_detect':
            self.models['yolo_corner_detect'] = self._load_yolo_corner_detection_model()
        elif model_name == 'text_corrector':
            self.models['text_corrector'] = self._load_text_correction_model()
        elif model_name == 'gemini_client':
            self.models['gemini_client'] = self._load_gemini_client()
