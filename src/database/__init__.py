"""
Database module for MongoDB operations.
"""

from .mongodb import MongoDBClient
from .models import OCRResult

__all__ = ["MongoDBClient", "OCRResult"]
