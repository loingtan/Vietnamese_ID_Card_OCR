"""
Tests for model manager functionality - Fixed version.
"""

from models.model_manager import ModelManager
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelManagerActual:
    """Test actual ModelManager functionality."""

    @patch('models.model_manager.torch.cuda.is_available')
    def test_model_manager_initialization_cpu(self, mock_cuda_available):
        """Test ModelManager initialization with CPU."""
        mock_cuda_available.return_value = False

        with patch.object(ModelManager, '_load_all_models') as mock_load:
            manager = ModelManager()
            assert manager.device == "cpu"
            mock_load.assert_called_once()

    def test_model_manager_with_config(self):
        """Test ModelManager with custom config."""
        mock_config = Mock()
        mock_config.google_ai_api_key = "test_key"

        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager(config=mock_config)
            assert manager.config == mock_config
            assert manager.api_key == "test_key"

    def test_get_device_returns_string(self):
        """Test get_device returns device string."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()
            device = manager.get_device()
            assert isinstance(device, str)
            assert device in ["cpu", "cuda"]

    def test_get_model_existing(self):
        """Test getting an existing model."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()
            mock_model = Mock()
            manager.models["test_model"] = mock_model

            result = manager.get_model("test_model")
            assert result == mock_model

    def test_get_model_not_existing(self):
        """Test getting a non-existing model."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()

            result = manager.get_model("non_existing_model")
            assert result is None

    def test_reload_model_existing(self):
        """Test reloading an existing model."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()
            mock_model = Mock()
            manager.models["vietocr"] = mock_model

            with patch.object(manager, '_load_vietocr_model') as mock_load:
                mock_new_model = Mock()
                mock_load.return_value = mock_new_model

                manager.reload_model("vietocr")
                assert manager.models["vietocr"] == mock_new_model

    def test_reload_model_non_existing(self):
        """Test reloading a non-existing model."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()

            # Should not raise an error
            manager.reload_model("non_existing_model")

    @patch('models.model_manager.torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True

        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()
            assert manager.device == "cuda"

    @patch('models.model_manager.torch.cuda.is_available')
    def test_device_selection_cuda_not_available(self, mock_cuda_available):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False

        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()
            assert manager.device == "cpu"


class TestModelManagerLoading:
    """Test model loading functionality."""

    @patch('vietocr.tool.predictor.Predictor')
    @patch('vietocr.tool.config.Cfg')
    @patch('streamlit.error')
    def test_load_vietocr_model(self, mock_st_error, mock_cfg, mock_predictor):
        """Test VietOCR model loading."""
        mock_config = Mock()
        mock_cfg.load_config_from_name.return_value = mock_config
        mock_model = Mock()
        mock_predictor.return_value = mock_model

        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager()
            # Mock the device attribute
            manager.device = "cpu"
            # Since we can't control the actual model loading, just verify it's not None
            model = manager._load_vietocr_model()
            assert model is not None
            mock_cfg.load_config_from_name.assert_called_once_with(
                'vgg_transformer')

    @patch('ultralytics.YOLO')
    @patch('streamlit.error')
    def test_load_yolo_text_detection_model(self, mock_st_error, mock_yolo):
        """Test YOLO text detection model loading."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model

        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager()
            manager.device = "cpu"
            with patch('pathlib.Path.exists', return_value=True):
                model = manager._load_yolo_text_detection_model()
                # Verify YOLO was called and model was moved to device
                mock_yolo.assert_called_once()
                mock_model.to.assert_called_once_with("cpu")

    @patch('ultralytics.YOLO')
    @patch('streamlit.error')
    def test_load_yolo_corner_detection_model(self, mock_st_error, mock_yolo):
        """Test YOLO corner detection model loading."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model

        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager()
            manager.device = "cpu"
            with patch('pathlib.Path.exists', return_value=True):
                model = manager._load_yolo_corner_detection_model()
                # Verify YOLO was called and result is a list containing the model
                mock_yolo.assert_called_once()
                mock_model.to.assert_called_once_with("cpu")
                assert isinstance(model, list)

    @patch('transformers.pipeline')
    @patch('streamlit.error')
    def test_load_text_correction_model(self, mock_st_error, mock_pipeline):
        """Test text correction model loading."""
        mock_model = Mock()
        mock_pipeline.return_value = mock_model

        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager()
            manager.device = "cpu"
            model = manager._load_text_correction_model()
            # Verify pipeline was called with correct parameters
            mock_pipeline.assert_called_once_with(
                "text2text-generation",
                model="bmd1905/vietnamese-correction-v2",
                device=-1  # CPU device
            )

    @patch('google.genai.Client')
    @patch('streamlit.error')
    def test_load_gemini_client_with_api_key(self, mock_st_error, mock_client):
        """Test Gemini client loading with API key."""
        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager(api_key="test_key")

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance

            client = manager._load_gemini_client()

            mock_client.assert_called_once_with(api_key="test_key")
            assert client == mock_client_instance

    def test_load_gemini_client_without_api_key(self):
        """Test Gemini client loading without API key."""
        with patch.object(ModelManager, '_load_all_models', return_value=None):
            manager = ModelManager()

            client = manager._load_gemini_client()
            assert client is None


class TestModelManagerIntegrationActual:
    """Integration tests for ModelManager."""

    @patch('models.model_manager.ModelManager._load_vietocr_model')
    @patch('models.model_manager.ModelManager._load_yolo_text_detection_model')
    @patch('models.model_manager.ModelManager._load_yolo_text_detection_model_v2')
    @patch('models.model_manager.ModelManager._load_yolo_corner_detection_model')
    @patch('models.model_manager.ModelManager._load_text_correction_model')
    def test_load_all_models_mock(self, mock_text_corr, mock_corner, mock_text_v2, mock_text, mock_vietocr):
        """Test loading all models with mocks."""
        # Setup mock return values
        mock_vietocr.return_value = Mock()
        mock_text.return_value = Mock()
        mock_text_v2.return_value = Mock()
        mock_corner.return_value = Mock()
        mock_text_corr.return_value = Mock()

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

    def test_model_manager_state(self):
        """Test ModelManager maintains state correctly."""
        with patch.object(ModelManager, '_load_all_models'):
            manager = ModelManager()

            # Add a model manually
            mock_model = Mock()
            manager.models["test_model"] = mock_model

            # Verify state is maintained
            assert manager.get_model("test_model") == mock_model
            assert len(manager.models) == 1

    def test_model_manager_error_handling(self):
        """Test ModelManager handles errors gracefully."""
        with patch.object(ModelManager, '_load_all_models', side_effect=Exception("Load failed")):
            with pytest.raises(Exception):
                ModelManager()
