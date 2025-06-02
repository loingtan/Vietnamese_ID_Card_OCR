"""
Simple configuration tests to verify test discovery works.
"""

from config import Config, get_config
import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_config():
    """Test basic configuration creation."""
    config = Config()
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 8080


def test_environment_override():
    """Test environment variable override."""
    with patch.dict(os.environ, {'API_PORT': '9000'}):
        config = Config()
        assert config.API_PORT == 9000


def test_get_config_function():
    """Test get_config function."""
    config = get_config()
    assert config is not None
    assert hasattr(config, 'API_HOST')
    assert hasattr(config, 'API_PORT')
