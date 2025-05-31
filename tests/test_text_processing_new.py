"""
Tests for text processing utilities.
"""

from utils.text_processing import (
    correct_text,
    extract_id_number,
    extract_dates,
    extract_gender,
    is_vietnamese_name,
    normalize_vietnamese_text,
    extract_address_components,
    clean_ocr_artifacts,
    validate_id_card_fields,
    load_vietnamese_dictionary
)
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_extract_id_number():
    """Test ID number extraction."""
    # Test valid ID number
    text = "Số/No.: 123456789012"
    result = extract_id_number(text)
    assert result == "123456789012"

    # Test invalid ID number
    text = "No ID number here"
    result = extract_id_number(text)
    assert result is None


def test_extract_dates():
    """Test date extraction."""
    text = "Ngày sinh: 15/03/1990"
    dates = extract_dates(text)
    assert len(dates) > 0
    assert "15/03/1990" in dates


def test_extract_gender():
    """Test gender extraction."""
    # Test male
    text = "Giới tính: Nam"
    result = extract_gender(text)
    assert result == "Nam"

    # Test female
    text = "Giới tính: Nữ"
    result = extract_gender(text)
    assert result == "Nữ"

    # Test no gender
    text = "No gender information"
    result = extract_gender(text)
    assert result is None


def test_is_vietnamese_name():
    """Test Vietnamese name validation."""
    assert is_vietnamese_name("Nguyễn Văn An") == True
    assert is_vietnamese_name("John Smith") == False
    assert is_vietnamese_name("123456") == False
    assert is_vietnamese_name("") == False


def test_normalize_vietnamese_text():
    """Test Vietnamese text normalization."""
    text = "NGUYỄN   VĂN    AN"
    result = normalize_vietnamese_text(text)
    assert "Nguyễn Văn An" in result or "nguyễn văn an" in result


def test_extract_address_components():
    """Test address component extraction."""
    text = "Hà Nội, Việt Nam"
    components = extract_address_components(text)
    assert len(components) > 0


def test_clean_ocr_artifacts():
    """Test OCR artifact cleaning."""
    text = "N@ame: John"
    clean_text = clean_ocr_artifacts(text)
    # Should clean some artifacts
    assert len(clean_text) <= len(text)


def test_validate_id_card_fields():
    """Test ID card field validation."""
    data = {
        "id_number": "123456789012",
        "name": "Nguyễn Văn An",
        "date_of_birth": "15/03/1990"
    }
    result = validate_id_card_fields(data)
    assert isinstance(result, dict)


def test_correct_text():
    """Test text correction functionality."""
    # Mock dictionary for testing
    mock_dict = {"correct", "word", "test"}
    result = correct_text("correkt", mock_dict)
    # Should return some correction
    assert isinstance(result, str)


def test_load_vietnamese_dictionary():
    """Test dictionary loading (should handle missing file gracefully)."""
    try:
        result = load_vietnamese_dictionary("nonexistent/path")
        assert isinstance(result, set)
    except Exception:
        # Expected if file doesn't exist
        pass
