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


try:
    from config.settings import get_config
    CONFIG_AVAILABLE = True
except (ImportError, Exception) as e:

    CONFIG_AVAILABLE = False
    get_config = None


class ModelManager:
    def __init__(self, api_key: str = None, config=None):
        try:
            if CONFIG_AVAILABLE and config is None and get_config is not None:
                self.config = get_config()
            else:
                self.config = config
        except Exception:
            self.config = config
        if self.config and hasattr(self.config, 'models') and hasattr(self.config.models, 'device') and self.config.models.device != "auto":
            self.device = self.config.models.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = Path(__file__).parent.parent.parent / "data" / "models"

        self.models = {}
        try:
            self.api_key = api_key or (self.config.google_ai_api_key if self.config and hasattr(
                self.config, 'google_ai_api_key') else None)
        except Exception:
            self.api_key = api_key

        self._load_all_models()

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

    def _load_yolo_text_detection_model(self):
        try:
            model_path = self.base_dir / "yolo_detect_text" / "best.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO text detection model: {e}")
            return None

    def _load_yolo_text_detection_model_v2(self):
        try:
            model_path = self.base_dir / "yolo_detect_text" / "bestv2.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO text detection model v2: {e}")
            return None

    def _load_yolo_corner_detection_model(self):
        try:
            model_path = self.base_dir / "corner_detection_model" / \
                "weight" / "29_03_25-YOLOv11n-Corner-best_metrics.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = YOLO(str(model_path))
            model.to(self.device)
            return [model]  # Return as list for compatibility
        except Exception as e:
            logger.error(f"Error loading YOLO corner detection model: {e}")
            return None

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
