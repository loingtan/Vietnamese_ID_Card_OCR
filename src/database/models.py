"""
Data models for MongoDB collections.
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass


@dataclass
class OCRResult:
    """OCR processing result data model."""

    session_id: str
    image_filename: str
    extracted_info: Dict[str, Any]
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None
    confidence_scores: Dict[str, float] = None
    detected_text_regions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.detected_text_regions is None:
            self.detected_text_regions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "session_id": self.session_id,
            "image_filename": self.image_filename,
            "extracted_info": self.extracted_info,
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "confidence_scores": self.confidence_scores,
            "detected_text_regions": self.detected_text_regions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        """Create instance from dictionary."""
        return cls(
            session_id=data["session_id"],
            image_filename=data["image_filename"],
            extracted_info=data["extracted_info"],
            processing_time=data["processing_time"],
            success=data.get("success", True),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp"),
            confidence_scores=data.get("confidence_scores", {}),
            detected_text_regions=data.get("detected_text_regions", [])
        )
