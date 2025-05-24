"""
Database module for MongoDB operations.
"""

from .mongodb import MongoDBClient
from .models import OCRResult, UserSession, ProcessingMetrics

__all__ = ["MongoDBClient", "OCRResult", "UserSession", "ProcessingMetrics"]
