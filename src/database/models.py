"""
Data models for MongoDB collections.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid


@dataclass
class OCRResult:
    """OCR processing result data model."""

    session_id: str
    image_filename: str
    extracted_info: Dict[str, Any]
    processing_time: float
    confidence_scores: Dict[str, float]
    detected_text_regions: List[Dict[str, Any]]
    qr_code_data: Optional[str] = None
    gemini_response: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "session_id": self.session_id,
            "image_filename": self.image_filename,
            "extracted_info": self.extracted_info,
            "processing_time": self.processing_time,
            "confidence_scores": self.confidence_scores,
            "detected_text_regions": self.detected_text_regions,
            "qr_code_data": self.qr_code_data,
            "gemini_response": self.gemini_response,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        """Create instance from dictionary."""
        return cls(
            session_id=data["session_id"],
            image_filename=data["image_filename"],
            extracted_info=data["extracted_info"],
            processing_time=data["processing_time"],
            confidence_scores=data["confidence_scores"],
            detected_text_regions=data["detected_text_regions"],
            qr_code_data=data.get("qr_code_data"),
            gemini_response=data.get("gemini_response"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp")
        )


@dataclass
class UserSession:
    """User session data model."""

    session_id: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    processed_images_count: int = 0
    total_processing_time: float = 0.0
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()

    def update_activity(self, processing_time: float = 0.0):
        """Update session activity."""
        self.last_activity = datetime.utcnow()
        if processing_time > 0:
            self.processed_images_count += 1
            self.total_processing_time += processing_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "session_id": self.session_id,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "processed_images_count": self.processed_images_count,
            "total_processing_time": self.total_processing_time,
            "settings": self.settings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Create instance from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_agent=data.get("user_agent"),
            ip_address=data.get("ip_address"),
            created_at=data.get("created_at"),
            last_activity=data.get("last_activity"),
            processed_images_count=data.get("processed_images_count", 0),
            total_processing_time=data.get("total_processing_time", 0.0),
            settings=data.get("settings", {})
        )


@dataclass
class ProcessingMetrics:
    """Processing metrics data model."""

    operation: str  # e.g., "ocr_processing", "text_detection", "text_recognition"
    processing_time: float
    success: bool
    session_id: Optional[str] = None
    image_size: Optional[tuple] = None
    model_used: Optional[str] = None
    confidence_score: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "operation": self.operation,
            "processing_time": self.processing_time,
            "success": self.success,
            "session_id": self.session_id,
            "image_size": self.image_size,
            "model_used": self.model_used,
            "confidence_score": self.confidence_score,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "additional_data": self.additional_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingMetrics":
        """Create instance from dictionary."""
        return cls(
            operation=data["operation"],
            processing_time=data["processing_time"],
            success=data["success"],
            session_id=data.get("session_id"),
            image_size=data.get("image_size"),
            model_used=data.get("model_used"),
            confidence_score=data.get("confidence_score"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp"),
            additional_data=data.get("additional_data", {})
        )


@dataclass
class IDCardInfo:
    """Structured ID Card information."""

    id_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    nationality: Optional[str] = None
    place_of_origin: Optional[str] = None
    place_of_residence: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    qr_code_data: Optional[str] = None
    card_type: Optional[str] = None  # "old" or "new"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id_number": self.id_number,
            "full_name": self.full_name,
            "date_of_birth": self.date_of_birth,
            "gender": self.gender,
            "nationality": self.nationality,
            "place_of_origin": self.place_of_origin,
            "place_of_residence": self.place_of_residence,
            "issue_date": self.issue_date,
            "expiry_date": self.expiry_date,
            "qr_code_data": self.qr_code_data,
            "card_type": self.card_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IDCardInfo":
        """Create instance from dictionary."""
        return cls(
            id_number=data.get("id_number"),
            full_name=data.get("full_name"),
            date_of_birth=data.get("date_of_birth"),
            gender=data.get("gender"),
            nationality=data.get("nationality"),
            place_of_origin=data.get("place_of_origin"),
            place_of_residence=data.get("place_of_residence"),
            issue_date=data.get("issue_date"),
            expiry_date=data.get("expiry_date"),
            qr_code_data=data.get("qr_code_data"),
            card_type=data.get("card_type")
        )

    def is_valid(self) -> bool:
        """Check if the ID card info has minimum required fields."""
        return bool(self.id_number and self.full_name)

    def get_completeness_score(self) -> float:
        """Get completeness score (0-1) based on filled fields."""
        total_fields = 11
        filled_fields = sum(1 for field in [
            self.id_number, self.full_name, self.date_of_birth,
            self.gender, self.nationality, self.place_of_origin,
            self.place_of_residence, self.issue_date, self.expiry_date,
            self.qr_code_data, self.card_type
        ] if field is not None and field != "")

        return filled_fields / total_fields
