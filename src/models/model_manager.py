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
import mlflow
from mlflow import artifacts
from src.config import Config


class ModelManager:
    """Manages all models used in the Vietnamese ID Card OCR system."""

    def __init__(self, api_key: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.api_key = api_key

        # Use MLflow config from config.py
        self.mlflow_enabled = Config.MLFLOW_ENABLED
        self.mlflow_tracking_uri = Config.MLFLOW_TRACKING_URI
        self.mlflow_model_artifacts = Config.get_mlflow_model_config()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Fallback paths for local weights
        self.local_weights = {
            "yolo_text_detect": str(Config.YOLO_TEXT_MODEL_PATH),
            "yolo_text_detect_v2": str(Config.YOLO_TEXT_V2_MODEL_PATH),
            "yolo_corner_detect": str(Config.YOLO_CORNER_MODEL_PATH)
        }

        self._load_all_models()

    def _download_yolo_weight_from_mlflow(self, model_key: str):
        """Download YOLO model weights from MLflow using run_id and artifact_path."""
        artifact_info = self.mlflow_model_artifacts.get(model_key.replace("_detect", ""), None)
        if not artifact_info:
            return None
        run_id = artifact_info.get("run_id")
        artifact_path = artifact_info.get("artifact_path")
        if not run_id or not artifact_path:
            return None
        try:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            return local_path
        except Exception as e:
            st.warning(f"Could not download {model_key} from MLflow: {e}")
            return None

    def _load_all_models(self):
        """Load all required models."""
        self.models['vietocr'] = self._load_vietocr_model()
        self.models['yolo_text_detect'] = self._load_yolo_model('yolo_text_detect')
        self.models['yolo_text_detect_v2'] = self._load_yolo_model('yolo_text_detect_v2')
        self.models['yolo_corner_detect'] = [self._load_yolo_model('yolo_corner_detect')]
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

    def _load_yolo_model(self, model_key: str):
        """Load YOLO model from MLflow run artifact or fallback to local file."""
        # Try MLflow first
        if self.mlflow_enabled:
            weight_path = self._download_yolo_weight_from_mlflow(model_key)
            if weight_path and Path(weight_path).exists():
                try:
                    model = YOLO(str(weight_path))
                    model.to(self.device)
                    return model
                except Exception as e:
                    st.warning(f"Could not load {model_key} from MLflow artifact: {e}")

        # Fallback to local file
        try:
            local_path = Path(self.local_weights.get(model_key, ""))
            if not local_path.exists():
                raise FileNotFoundError(f"Local fallback weight not found: {local_path}")
            model = YOLO(str(local_path))
            model.to(self.device)
            return model
        except Exception as e:
            st.error(f"Failed to load {model_key} from both MLflow and local fallback: {e}")
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
        elif model_name in self.local_weights:
            model = self._load_yolo_model(model_name)
            if model_name == 'yolo_corner_detect':
                self.models[model_name] = [model]
            else:
                self.models[model_name] = model
        elif model_name == 'text_corrector':
            self.models['text_corrector'] = self._load_text_correction_model()
        elif model_name == 'gemini_client':
            self.models['gemini_client'] = self._load_gemini_client()
