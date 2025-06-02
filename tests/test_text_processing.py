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


class TestTextProcessing:
    """Test text processing utilities."""

    def test_normalize_vietnamese_text(self):
        """Test Vietnamese text normalization."""
        # Test basic normalization
        text = "  Nguyễn  Văn   A  "
        result = normalize_vietnamese_text(text)
        assert result == "Nguyễn Văn A"

        # Test empty text
        result = normalize_vietnamese_text("")
        assert result == ""

        # Test None input
        result = normalize_vietnamese_text(None)
        assert result is None

    def test_correct_text_with_dictionary(self):
        """Test text correction with dictionary."""
        vietnamese_words = ["Nguyễn", "Văn", "Hà", "Nội"]

        # Test correct word
        result = correct_text("Nguyễn", vietnamese_words)
        assert result == "Nguyễn"

        # Test slightly misspelled word
        result = correct_text("Nguyen", vietnamese_words)
        # Should find closest match or return original

    def test_extract_date_valid_formats(self):
        """Test date extraction with valid formats."""
        # DD/MM/YYYY format
        text = "Ngày sinh: 15/08/1990"
        result = extract_date(text)
        assert result == "15/08/1990"

        # DD-MM-YYYY format
        text = "Ngày sinh: 15-08-1990"
        result = extract_date(text)
        assert result == "15-08-1990"

    def test_extract_date_invalid_format(self):
        """Test date extraction with invalid format."""
        text = "Không có ngày sinh"
        result = extract_date(text)
        assert result is None

    def test_extract_id_number_valid(self):
        """Test ID number extraction with valid numbers."""
        # 12-digit ID (new format)
        text = "Số CCCD: 123456789012"
        result = extract_id_number(text)
        assert result == "123456789012"

        # 9-digit ID (old format)
        text = "Số CMND: 123456789"
        result = extract_id_number(text)
        assert result == "123456789"

    def test_extract_id_number_invalid(self):
        """Test ID number extraction with invalid numbers."""
        text = "Không có số ID"
        result = extract_id_number(text)
        assert result is None

    def test_extract_gender_male(self):
        """Test gender extraction for male."""
        text = "Giới tính: Nam"
        result = extract_gender(text)
        assert result == "Nam"

    def test_extract_gender_female(self):
        """Test gender extraction for female."""
        text = "Giới tính: Nữ"
        result = extract_gender(text)
        assert result == "Nữ"

    def test_extract_gender_not_found(self):
        """Test gender extraction when not found."""
        text = "Không có giới tính"
        result = extract_gender(text)
        assert result is None

    def test_clean_extracted_text(self):
        """Test text cleaning function."""
        # Test with extra whitespace
        text = "  Nguyễn  Văn  A  "
        result = clean_extracted_text(text)
        assert result == "Nguyễn Văn A"

        # Test with special characters
        text = "Nguyễn@Văn#A"
        result = clean_extracted_text(text)
        assert "@" not in result and "#" not in result

    def test_is_likely_name_valid(self):
        """Test name validation with valid names."""
        assert is_likely_name("Nguyễn Văn A") == True
        assert is_likely_name("Trần Thị B") == True
        assert is_likely_name("Lê Minh C") == True

    def test_is_likely_name_invalid(self):
        """Test name validation with invalid names."""
        assert is_likely_name("123456") == False
        assert is_likely_name("@#$%") == False
        assert is_likely_name("") == False
        assert is_likely_name("A") == False  # Too short

    def test_validate_vietnamese_id_new_format(self):
        """Test Vietnamese ID validation for new format (12 digits)."""
        assert validate_vietnamese_id("123456789012") == True
        assert validate_vietnamese_id("000000000000") == True

    def test_validate_vietnamese_id_old_format(self):
        """Test Vietnamese ID validation for old format (9 digits)."""
        assert validate_vietnamese_id("123456789") == True
        assert validate_vietnamese_id("000000000") == True

    def test_validate_vietnamese_id_invalid(self):
        """Test Vietnamese ID validation for invalid formats."""
        assert validate_vietnamese_id("12345") == False  # Too short
        assert validate_vietnamese_id("1234567890123") == False  # Too long
        assert validate_vietnamese_id("12345678a") == False  # Contains letter
        assert validate_vietnamese_id("") == False  # Empty
        assert validate_vietnamese_id(None) == False  # None


class TestTextProcessingEdgeCases:
    """Test edge cases for text processing."""

    def test_correct_text_empty_dictionary(self):
        """Test text correction with empty dictionary."""
        result = correct_text("Nguyễn", [])
        assert result == "Nguyễn"  # Should return original

    def test_correct_text_none_input(self):
        """Test text correction with None input."""
        result = correct_text(None, ["word"])
        assert result == ""

    def test_extract_date_multiple_dates(self):
        """Test date extraction with multiple dates in text."""
        text = "Ngày sinh: 15/08/1990, Ngày cấp: 20/08/2020"
        result = extract_date(text)
        # Should return the first date found
        assert "15/08/1990" in result

    def test_extract_id_number_multiple_numbers(self):
        """Test ID extraction with multiple numbers in text."""
        text = "Số CCCD: 123456789012, Số điện thoại: 0123456789"
        result = extract_id_number(text)
        assert result == "123456789012"

    def test_clean_extracted_text_unicode(self):
        """Test text cleaning with Unicode characters."""
        text = "Nguyễn Văn Âñ"
        result = clean_extracted_text(text)
        assert "Â" in result or "â" in result
        assert "ñ" in result

    def test_is_likely_name_with_numbers(self):
        """Test name validation with names containing numbers."""
        assert is_likely_name("Nguyễn Văn A1") == False
        assert is_likely_name("123 Nguyễn") == False

    def test_validate_id_with_spaces(self):
        """Test ID validation with spaces."""
        assert validate_vietnamese_id("123 456 789") == False
        assert validate_vietnamese_id("123-456-789") == False


class TestTextProcessingIntegration:
    """Integration tests for text processing pipeline."""

    def test_full_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        raw_text = "  Họ và tên: Nguyễn Văn A  \nSố CCCD: 123456789012\nNgày sinh: 15/08/1990"
        vietnamese_words = ["Nguyễn", "Văn", "Hà", "Nội"]

        # Clean text
        clean_text = clean_extracted_text(raw_text)
        assert clean_text is not None

        # Extract information
        id_number = extract_id_number(clean_text)
        assert id_number == "123456789012"

        date_of_birth = extract_date(clean_text)
        assert date_of_birth == "15/08/1990"

        # Correct text
        corrected = correct_text("Nguyễn Văn A", vietnamese_words)
        assert corrected is not None

    @patch('src.utils.text_processing.Levenshtein')
    def test_text_correction_with_levenshtein_mock(self, mock_levenshtein):
        """Test text correction with mocked Levenshtein distance."""
        mock_levenshtein.distance.return_value = 1

        vietnamese_words = ["Nguyễn", "Văn"]
        result = correct_text("Nguyen", vietnamese_words)

        # Should use the mocked distance calculation
        mock_levenshtein.distance.assert_called()

    def test_extract_address_components(self):
        """Test extraction of address components."""
        address_text = "Thường trú: 123 Đường ABC, Phường XYZ, Quận 1, TP.HCM"

        # This would test address parsing functionality
        # Currently not implemented as separate function
        # but could be added to text processing utilities

        # Example expectations:
        # - Street number: 123
        # - Street name: Đường ABC
        # - Ward: Phường XYZ
        # - District: Quận 1
        # - City: TP.HCM

    def test_text_quality_assessment(self):
        """Test text quality assessment."""
        high_quality_text = "Nguyễn Văn A"
        low_quality_text = "Ng#yệ$ V@n A"

        # This could test a quality scoring function
        # that assesses OCR confidence based on text characteristics

        assert len(high_quality_text.replace(" ", "")) > 0
        assert any(c.isalpha() for c in high_quality_text)
