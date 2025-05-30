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
import mlflow
from src.config import get_config

class ModelManager:
    """Manages all models used in the Vietnamese ID Card OCR system."""

    def __init__(self, api_key: str = None):
        self.config = get_config()
        self.device = "cuda" if torch.cuda.is_available() and not self.config.FORCE_CPU else "cpu"
        self.models = {}
        self.api_key = api_key
        self.local_weights = self.config.get_model_paths()
        
        if self.config.MLFLOW_ENABLED:
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            
        self._load_all_models()

    def _get_model_weights_path(self, model_key: str, download_dir: str = "./.temp", model_version: str = "1") -> str:
        """
        Get model weights path, trying MLflow first then falling back to local.
        
        Args:
            model_key (str): Key identifying the model (e.g., 'yolo_text', 'yolo_corner')
            download_dir (str): Directory to store downloaded weights, defaults to './.temp'
            model_version (str): Version of the model to download, defaults to 1
            
        Returns:
            str: Path to model weights file
            
        Raises:
            FileNotFoundError: If no weights found in MLflow or locally
        """
        # Create model-specific download directory to avoid conflicts
        model_download_dir = Path(download_dir) / model_key
        model_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Try MLflow first if enabled
        if self.config.MLFLOW_ENABLED:
            client = mlflow.tracking.MlflowClient()
            try:
                # Get artifact info from config
                artifact_info = self.config.MLFLOW_MODEL_ARTIFACTS.get(model_key)
                if artifact_info:
                    version = artifact_info.get("version", model_version)
                    artifact_path = artifact_info.get("artifact_path", "").rstrip("/")
                    
                    # Get run ID from model version
                    model_version = client.get_model_version(model_key, version)
                    version_run_id = model_version.run_id
                    print(f"Found run ID {version_run_id} for {model_key} version {version}")
                    
                    # Download using run ID and artifact path to model-specific directory
                    download_path = mlflow.artifacts.download_artifacts(
                        run_id=version_run_id,
                        artifact_path=artifact_path,
                        dst_path=str(model_download_dir)
                    )
                    
                    # Handle both file and directory downloads
                    download_path = Path(download_path)
                    if download_path.is_dir():
                        # Search for the specific model file
                        expected_name = Path(artifact_path).name
                        if expected_name:
                            model_file = download_path / expected_name
                            if model_file.exists() and model_file.suffix == '.pt':
                                print(f"Using MLflow weights: {model_file}")
                                return str(model_file)
                        
                        # Fallback to searching for any .pt file
                        pt_files = list(download_path.glob("*.pt"))
                        if pt_files:
                            model_path = str(pt_files[0])
                            print(f"Using MLflow weights from directory: {model_path}")
                            return model_path
                    elif download_path.suffix == '.pt':
                        print(f"Using MLflow weights: {download_path}")
                        return str(download_path)
                    
            except Exception as e:
                print(f"MLflow download failed for {model_key}: {e}")

        # Fallback to local weights
        local_path = Path(self.local_weights[model_key])
        if local_path.exists():
            print(f"Using local weights: {local_path}")
            return str(local_path)
        
        raise FileNotFoundError(f"No weights found for {model_key} in MLflow or locally")

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
            model_path = _self._get_model_weights_path("yolo_text")
            model = YOLO(model_path)
            model.to(_self.device)
            return model
        except Exception as e:
            st.error(f"Error loading YOLO text detection model: {e}")
            return None

    @st.cache_resource
    def _load_yolo_text_detection_model_v2(_self):
        """Load YOLO v2 model for text detection."""
        try:
            model_path = _self._get_model_weights_path("yolo_text_v2")
            model = YOLO(model_path)
            model.to(_self.device)
            return model
        except Exception as e:
            st.error(f"Error loading YOLO text detection model v2: {e}")
            return None

    @st.cache_resource
    def _load_yolo_corner_detection_model(_self):
        """Load YOLO model for ID card corner detection."""
        try:
            model_path = _self._get_model_weights_path("yolo_corner")
            model = YOLO(model_path)
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