"""
Working tests for model manager functionality.
"""

from models.model_manager import ModelManager
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelManagerBasic:
    """Test basic ModelManager functionality."""

    @patch.object(ModelManager, '_load_all_models')
    def test_model_manager_initialization(self, mock_load_all):
        """Test ModelManager initialization."""
        mock_load_all.return_value = None
        manager = ModelManager()

        assert hasattr(manager, 'device')
        assert hasattr(manager, 'models')
        assert hasattr(manager, 'api_key')
        mock_load_all.assert_called_once()

    @patch.object(ModelManager, '_load_all_models')
    def test_get_device_returns_string(self, mock_load_all):
        """Test get_device returns device string."""
        mock_load_all.return_value = None
        manager = ModelManager()

        device = manager.get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda"]

    @patch.object(ModelManager, '_load_all_models')
    def test_get_model_existing(self, mock_load_all):
        """Test getting an existing model."""
        mock_load_all.return_value = None
        manager = ModelManager()

        mock_model = Mock()
        manager.models["test_model"] = mock_model

        result = manager.get_model("test_model")
        assert result == mock_model

    @patch.object(ModelManager, '_load_all_models')
    def test_get_model_not_existing(self, mock_load_all):
        """Test getting a non-existing model."""
        mock_load_all.return_value = None
        manager = ModelManager()

        result = manager.get_model("non_existing_model")
        assert result is None

    @patch.object(ModelManager, '_load_all_models')
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda_available, mock_load_all):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_load_all.return_value = None

        manager = ModelManager()
        assert manager.device == "cuda"

    @patch.object(ModelManager, '_load_all_models')
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_not_available(self, mock_cuda_available, mock_load_all):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_load_all.return_value = None

        manager = ModelManager()
        assert manager.device == "cpu"

    @patch.object(ModelManager, '_load_all_models')
    def test_model_manager_with_api_key(self, mock_load_all):
        """Test ModelManager with API key."""
        mock_load_all.return_value = None

        manager = ModelManager(api_key="test_key")
        assert manager.api_key == "test_key"

    @patch.object(ModelManager, '_load_all_models')
    def test_model_manager_with_config(self, mock_load_all):
        """Test ModelManager with custom config."""
        mock_load_all.return_value = None
        mock_config = Mock()
        mock_config.google_ai_api_key = "config_key"

        manager = ModelManager(config=mock_config)
        assert manager.config == mock_config


class TestModelManagerReload:
    """Test model reloading functionality."""

    @patch.object(ModelManager, '_load_all_models')
    @patch.object(ModelManager, '_load_vietocr_model')
    def test_reload_vietocr_model(self, mock_load_vietocr, mock_load_all):
        """Test reloading VietOCR model."""
        mock_load_all.return_value = None
        mock_new_model = Mock()
        mock_load_vietocr.return_value = mock_new_model

        manager = ModelManager()
        manager.reload_model("vietocr")

        mock_load_vietocr.assert_called_once()
        assert manager.models["vietocr"] == mock_new_model

    @patch.object(ModelManager, '_load_all_models')
    @patch.object(ModelManager, '_load_yolo_text_detection_model')
    def test_reload_yolo_text_model(self, mock_load_yolo, mock_load_all):
        """Test reloading YOLO text model."""
        mock_load_all.return_value = None
        mock_new_model = Mock()
        mock_load_yolo.return_value = mock_new_model

        manager = ModelManager()
        manager.reload_model("yolo_text_detect")

        mock_load_yolo.assert_called_once()
        assert manager.models["yolo_text_detect"] == mock_new_model

    @patch.object(ModelManager, '_load_all_models')
    def test_reload_non_existing_model(self, mock_load_all):
        """Test reloading a non-existing model type."""
        mock_load_all.return_value = None
        manager = ModelManager()

        # Should not raise an error
        manager.reload_model("non_existing_model")


class TestModelManagerIntegration:
    """Integration tests for ModelManager."""

    @patch.object(ModelManager, '_load_vietocr_model')
    @patch.object(ModelManager, '_load_yolo_text_detection_model')
    @patch.object(ModelManager, '_load_yolo_text_detection_model_v2')
    @patch.object(ModelManager, '_load_yolo_corner_detection_model')
    @patch.object(ModelManager, '_load_text_correction_model')
    @patch.object(ModelManager, '_load_gemini_client')
    def test_load_all_models_called(
        self,
        mock_gemini,
        mock_text_corr,
        mock_corner,
        mock_text_v2,
        mock_text,
        mock_vietocr
    ):
        """Test that all model loading methods are called."""
        # Setup mock return values
        mock_vietocr.return_value = Mock()
        mock_text.return_value = Mock()
        mock_text_v2.return_value = Mock()
        mock_corner.return_value = Mock()
        mock_text_corr.return_value = Mock()
        mock_gemini.return_value = None  # No API key

        manager = ModelManager()

        # Verify all models are loaded
        assert 'vietocr' in manager.models
        assert 'yolo_text_detect' in manager.models
        assert 'yolo_text_detect_v2' in manager.models
        assert 'yolo_corner_detect' in manager.models
        assert 'text_corrector' in manager.models

        # Verify all load methods were called
        mock_vietocr.assert_called_once()
        mock_text.assert_called_once()
        mock_text_v2.assert_called_once()
        mock_corner.assert_called_once()
        mock_text_corr.assert_called_once()

    @patch.object(ModelManager, '_load_all_models')
    def test_model_state_persistence(self, mock_load_all):
        """Test ModelManager maintains state correctly."""
        mock_load_all.return_value = None
        manager = ModelManager()

        # Add models manually
        mock_model1 = Mock()
        mock_model2 = Mock()
        manager.models["model1"] = mock_model1
        manager.models["model2"] = mock_model2

        # Verify state is maintained
        assert manager.get_model("model1") == mock_model1
        assert manager.get_model("model2") == mock_model2
        assert len(manager.models) == 2

    @patch.object(ModelManager, '_load_all_models')
    def test_model_manager_error_handling(self, mock_load_all):
        """Test ModelManager handles initialization errors gracefully."""
        mock_load_all.side_effect = Exception("Load failed")

        with pytest.raises(Exception, match="Load failed"):
            ModelManager()


class TestModelManagerMocking:
    """Test model loading with proper mocking."""

    @patch('streamlit.error')
    @patch('vietocr.tool.predictor.Predictor')
    @patch('vietocr.tool.config.Cfg')
    def test_vietocr_loading_mocked(self, mock_cfg, mock_predictor, mock_st_error):
        """Test VietOCR loading behavior without actual model loading."""
        mock_config = Mock()
        mock_cfg.load_config_from_name.return_value = mock_config
        mock_model = Mock()
        mock_predictor.return_value = mock_model

        # Patch the cache decorator to bypass caching
        with patch('streamlit.cache_resource', lambda x: x):
            with patch.object(ModelManager, '_load_all_models'):
                manager = ModelManager()
                manager.device = "cpu"

                # Test the actual loading method
                result = manager._load_vietocr_model()

                # Verify the config was loaded and predictor was created
                mock_cfg.load_config_from_name.assert_called_with(
                    'vgg_transformer')
                assert mock_config['device'] == "cpu"
                mock_predictor.assert_called_with(mock_config)

    @patch('streamlit.error')
    @patch('pathlib.Path.exists')
    @patch('ultralytics.YOLO')
    def test_yolo_loading_mocked(self, mock_yolo, mock_path_exists, mock_st_error):
        """Test YOLO loading behavior without actual model loading."""
        mock_path_exists.return_value = True
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model

        # Patch the cache decorator to bypass caching
        with patch('streamlit.cache_resource', lambda x: x):
            with patch.object(ModelManager, '_load_all_models'):
                manager = ModelManager()
                manager.device = "cpu"

                # Test the actual loading method
                result = manager._load_yolo_text_detection_model()

                # Verify YOLO was called and model was moved to device
                mock_yolo.assert_called_once()
                mock_model.to.assert_called_with("cpu")

    @patch('streamlit.error')
    @patch('transformers.pipeline')
    def test_text_correction_loading_mocked(self, mock_pipeline, mock_st_error):
        """Test text correction model loading behavior."""
        mock_model = Mock()
        mock_pipeline.return_value = mock_model

        # Patch the cache decorator to bypass caching
        with patch('streamlit.cache_resource', lambda x: x):
            with patch.object(ModelManager, '_load_all_models'):
                manager = ModelManager()
                manager.device = "cpu"

                # Test the actual loading method
                result = manager._load_text_correction_model()

                # Verify pipeline was called with correct parameters
                mock_pipeline.assert_called_with(
                    "text2text-generation",
                    model="bmd1905/vietnamese-correction-v2",
                    device=-1  # CPU device
                )
