"""
Model loading and management utilities for Vietnamese ID Card OCR.
"""

import torch
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import pipeline
from google import genai
from pathlib import Path
import os

import logging
logger = logging.getLogger(__name__)

import mlflow
from src.config import get_config

# try:
#     from config.settings import get_config
#     CONFIG_AVAILABLE = True
# except (ImportError, Exception) as e:

#     CONFIG_AVAILABLE = False
#     get_config = None


class ModelManager:
    def __init__(self, api_key: str = None, config=None):
        """Initialize ModelManager with optional API key and config.
        
        Args:
            api_key (str, optional): Google AI API key. Defaults to None.
            config (Config, optional): Configuration object. Defaults to None.
        """
        # Load configuration
        try:
            from src.config import Config, get_config
            self.config = config if config is not None else get_config()
            if not isinstance(self.config, Config):
                raise TypeError(f"Config must be instance of Config, got {type(self.config)}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError("Could not initialize model manager: configuration error")

        # Set device
        self.device = ("cuda" if torch.cuda.is_available() 
                      and not getattr(self.config, 'FORCE_CPU', False) 
                      else "cpu")
        
        # Initialize models dict and get paths
        self.models = {}
        # self.local_weights = self.config.get_model_paths()
        self.local_weights = self.config.get_train_model_paths()
        
        # Set API key
        self.api_key = api_key or getattr(self.config, 'GEMINI_API_KEY', None)
        
        # Setup MLflow if enabled
        if getattr(self.config, 'MLFLOW_ENABLED', False):
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        
        # Load all models
        self._load_all_models()

    # def _load_config(self, config):
    #     """Load configuration from provided config or get_config()."""
    #     try:
    #         if CONFIG_AVAILABLE and config is None and get_config is not None:
    #             return get_config()
    #         return config
    #     except Exception as e:
    #         logger.warning(f"Error loading config: {e}")
    #         return config

    # def _setup_device(self):
    #     """Setup device based on config and available hardware."""
    #     if (self.config and 
    #         hasattr(self.config, 'models') and 
    #         hasattr(self.config.models, 'device') and 
    #         self.config.models.device != "auto"):
    #         return self.config.models.device
        
    #     return "cuda" if torch.cuda.is_available() and not self.config.FORCE_CPU else "cpu"

    # def _setup_api_key(self, api_key):
    #     """Setup API key from provided key or config."""
    #     try:
    #         return api_key or (
    #             self.config.google_ai_api_key 
    #             if self.config and hasattr(self.config, 'google_ai_api_key') 
    #             else None
    #         )
    #     except Exception as e:
    #         logger.warning(f"Error setting up API key: {e}")
    #         return api_key

    def _get_model_weights_path(self, model_key: str, download_dir: str = "./.temp/serving", model_version: str = "1") -> str:
        """Get model weights path, trying MLflow first then falling back to local."""
        # Create model-specific download directory
        model_download_dir = Path(download_dir) / model_key
        model_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Try MLflow first if enabled
        if self.config.MLFLOW_ENABLED:
            client = mlflow.tracking.MlflowClient()
            try:
                artifact_info = self.config.MLFLOW_MODEL_ARTIFACTS.get(model_key)
                if artifact_info:
                    version = artifact_info.get("version", model_version)
                    artifact_path = artifact_info.get("artifact_path", "").rstrip("/")
                    
                    # Get run ID from model version
                    model_version = client.get_model_version(model_key, version)
                    version_run_id = model_version.run_id
                    print(f"Found run ID {version_run_id} for {model_key} version {version}")
                    
                    # Download using run ID and artifact path
                    download_path = mlflow.artifacts.download_artifacts(
                        run_id=version_run_id,
                        artifact_path=artifact_path,
                        dst_path=str(model_download_dir)
                    )
                    
                    # Handle both file and directory downloads
                    download_path = Path(download_path)
                    if download_path.is_dir():
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

        # Simple fallback to local weights
        try:
            local_path = self.local_weights[model_key]
            if not local_path:
                raise FileNotFoundError(f"No local path configured for {model_key}\n")
            
            local_path = Path(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Local weights not found at: {local_path}\n")
            if local_path.suffix != '.pt':
                raise ValueError(f"Local weights file is not a .pt file: {local_path}")
                
            print(f"Using local weights: {local_path}")
            return str(local_path)
            
        except Exception as e:
            raise FileNotFoundError(f"No valid weights found for {model_key}: {e}")

    def _load_all_models(self):
        self.models['vietocr'] = self._load_vietocr_model()
        self.models['yolo_text_detect'] = self._load_yolo_text_detection_model()
        self.models['yolo_text_detect_v2'] = self._load_yolo_text_detection_model_v2()
        self.models['yolo_corner_detect'] = self._load_yolo_corner_detection_model()
        self.models['text_corrector'] = self._load_text_correction_model()

        if self.api_key:
            self.models['gemini_client'] = self._load_gemini_client()

    def _load_vietocr_model(self):
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = True
            config['device'] = self.device
            predictor = Predictor(config)
            return predictor
        except Exception as e:
            logger.error(f"Error loading VietOCR model: {e}")
            return None

    def _load_yolo_model(self, model_key: str, task: str = "detect") -> YOLO:
        """Generic YOLO model loader that handles both PT and ONNX formats.
        
        Args:
            model_key (str): Key for model lookup
            task (str, optional): YOLO task type. Defaults to "detect".
        """
        try:
            model_path = Path(self._get_model_weights_path(model_key))
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # If PT file, convert to ONNX first
            if model_path.suffix == '.pt':
                onnx_path = model_path.with_suffix('.onnx')
                if not onnx_path.exists():
                    logger.info(f"Converting {model_path} to ONNX format...")
                    model = YOLO(str(model_path), task=task)
                    model.export(
                        format='onnx',
                        simplify=True,
                        dynamic=True,
                        # half=True,
                        device=self.device
                    )
                    logger.info(f"Model converted and saved to: {onnx_path}")
                model_path = onnx_path

            # Initialize YOLO with ONNX model
            model = YOLO(str(model_path), task=task)
            logger.info(f"Loaded YOLO model from: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading YOLO model {model_key}: {e}")
            return None

    def _load_yolo_text_detection_model(self):
        """Load YOLO text detection model."""
        return self._load_yolo_model("yolo_text_detect", task="detect")

    def _load_yolo_text_detection_model_v2(self):
        """Load YOLO text detection model v2."""
        return self._load_yolo_model("yolo_text_detect_v2", task="detect")

    def _load_yolo_corner_detection_model(self):
        """Load YOLO corner detection model."""
        model = self._load_yolo_model("yolo_corner_detect", task="detect")
        return [model] if model else None  # Return as list for compatibility

    def _load_text_correction_model(self):
        try:
            corrector = pipeline(
                "text2text-generation",
                model="bmd1905/vietnamese-correction-v2",
                device=0 if self.device == "cuda" else -1
            )
            return corrector
        except Exception as e:
            logger.error(f"Error loading text correction model: {e}")
            return None

    def _load_gemini_client(self):
        """Load Gemini client for AI processing."""
        try:
            if not self.api_key:
                return None
            client = genai.Client(api_key=self.api_key)
            return client
        except Exception as e:
            logger.error(f"Error loading Gemini client: {e}")
            return None

    def get_model(self, model_name: str):
        return self.models.get(model_name)

    def get_device(self):

        return self.device

    def reload_model(self, model_name: str):

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