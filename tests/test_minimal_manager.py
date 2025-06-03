"""
Minimal test version of ModelManager to isolate the issue.
"""

import torch
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

print("DEBUG: Starting model_manager execution")

# Import the new configuration system
try:
    from config.settings import get_config
    CONFIG_AVAILABLE = True
    print("DEBUG: Config import successful")
except (ImportError, Exception) as e:
    # Fallback for legacy imports or any other error
    CONFIG_AVAILABLE = False
    get_config = None
    print(f"DEBUG: Config import failed: {e}")

print("DEBUG: About to define ModelManager class")


class ModelManager:
    """Manages all models used in the Vietnamese ID Card OCR system."""

    def __init__(self, api_key: str = None, config=None):
        print("DEBUG: ModelManager.__init__ called")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = Path(__file__).parent.parent.parent / "data" / "models"
        self.models = {}
        self.api_key = api_key

    def get_device(self):
        """Get the current device (cuda/cpu)."""
        return self.device


print("DEBUG: ModelManager class defined")
print(f"DEBUG: ModelManager in globals: {'ModelManager' in globals()}")
