"""
Text processing and validation utilities for Vietnamese ID Card OCR.
"""

import re
from typing import List, Set, Optional
from pathlib import Path
from langdetect import detect, LangDetectException
from Levenshtein import distance as levenshtein_distance

# Try to import config, fallback to a simple config if not available
try:
    from ..config import get_config
    config = get_config()
except ImportError:
    # Fallback for testing
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def load_vietnamese_dictionary(dict_path: str = r'dictionary\dictionaries\hongocduc\words.txt') -> Set[str]:
    """
    Load Vietnamese dictionary for text correction.

    Args:
        dict_path: Path to the dictionary file

    Returns:
        Set of Vietnamese words
    """
    try:
        dict_file = Path(dict_path)
        if not dict_file.exists():
            print(f"Warning: Dictionary file not found at {dict_path}")
            return set()

        with open(dict_file, 'r', encoding='utf-8') as file:
            words = {line.strip() for line in file if line.strip()}
        return words
    except Exception as e:
        print(f"Error loading Vietnamese dictionary: {e}")
        return set()


def correct_text(text: str, candidates: Set[str], threshold: int = 2) -> str:
    """
    Correct text using Levenshtein distance against dictionary candidates.

    Args:
        text: Text to correct
        candidates: Set of candidate words
        threshold: Maximum edit distance for correction

    Returns:
        Corrected text or original if no close match found
    """
    if not candidates or not text:
        return text

    closest_match = min(
        candidates, key=lambda c: levenshtein_distance(text, c))
    if levenshtein_distance(text, closest_match) <= threshold:
        return closest_match
    return text


def safe_detect(text: str) -> Optional[str]:
    """
    Safely detect language of text with error handling.

    Args:
        text: Text to analyze

    Returns:
        Language code or None if detection fails
    """
    try:
        if len(text.strip()) < 3:
            return None
        return detect(text)
    except LangDetectException:
        return None


def extract_id_number(text: str) -> Optional[str]:
    """
    Extract Vietnamese ID number from text (9 or 12 digits).

    Args:
        text: Text to search

    Returns:
        ID number if found, None otherwise
    """
    # Pattern for 9 or 12 digit ID numbers
    id_pattern = re.compile(r'\b\d{9}(?:\d{3})?\b')
    match = id_pattern.search(text)
    return match.group(0) if match else None


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text in various Vietnamese formats.

    Args:
        text: Text to search

    Returns:
        List of found dates
    """
    dates = []

    # Standard DD/MM/YYYY format
    date_pattern = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')
    dates.extend(date_pattern.findall(text))

    # Alternative formats: DD.MM.YYYY, DD-MM-YYYY, YYYY-MM-DD
    alt_patterns = [
        r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b'
    ]

    for pattern in alt_patterns:
        dates.extend(re.findall(pattern, text))

    return dates


def extract_gender(text: str) -> Optional[str]:
    """
    Extract gender from Vietnamese text.

    Args:
        text: Text to analyze

    Returns:
        'Nam' for male, 'Nữ' for female, None if not found
    """
    text_lower = text.lower()

    # Check female terms first to avoid 'female' matching 'male'
    female_terms = ['nữ', 'female', 'giới tính: nữ']
    for term in female_terms:
        if term in text_lower:
            return "Nữ"

    male_terms = ['nam', 'male', 'giới tính: nam']
    for term in male_terms:
        if term in text_lower:
            return "Nam"

    return None


def is_vietnamese_name(text: str) -> bool:
    """
    Check if text appears to be a Vietnamese name.

    Args:
        text: Text to check

    Returns:
        True if likely a Vietnamese name
    """
    if not text or len(text) < 3:
        return False

    # Vietnamese names are typically:
    # - Uppercase
    # - Multiple words (at least 2)
    # - No digits
    # - Reasonable length
    return (text.isupper() and
            len(text.split()) >= 2 and
            not any(c.isdigit() for c in text) and
            5 <= len(text) <= 50)


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text for better processing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return text

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Fix common OCR mistakes in Vietnamese
    replacements = {
        'ố': 'ố',  # Ensure proper diacritics
        'ề': 'ề',
        'ệ': 'ệ',
        'ộ': 'ộ',
        'ủ': 'ủ',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def extract_address_components(text: str) -> List[str]:
    """
    Extract address components from Vietnamese text.

    Args:
        text: Text containing address information

    Returns:
        List of address components
    """
    # Patterns for Vietnamese address components
    address_patterns = [
        r"(Số|số|Nhà|nhà)\s+\d+",  # House numbers
        r"(Tổ|Tó)\s+\d+",          # Group numbers
        r"(Thon|thon|Thôn|thôn)\s+\w+",  # Villages
        r"(Khu|khu|Khối|khối|kp|Kp)\s+\d+",  # Districts/blocks
        # Administrative units
        r"(Xã|Phường|Thị trấn|Huyện|Quận|Thành phố|Tỉnh)\s+\w+",
        r"(Đường|đường|Phố|phố)\s+\w+"  # Streets
    ]

    components = []
    for pattern in address_patterns:
        matches = re.findall(pattern, text)
        components.extend([match if isinstance(match, str)
                          else ' '.join(match) for match in matches])

    return components


def clean_ocr_artifacts(text: str) -> str:
    """
    Clean common OCR artifacts from text.

    Args:
        text: Text with potential OCR errors

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Remove common OCR artifacts
    artifacts = ['|', '_', '~', '`', '^']
    for artifact in artifacts:
        text = text.replace(artifact, '')

    # Fix common character confusions
    replacements = {
        '0': 'O',  # In names, 0 is likely O
        '1': 'I',  # In some contexts
        '5': 'S',  # Common confusion
    }

    # Apply replacements only if text doesn't contain numbers (likely a name)
    if not any(c.isdigit() for c in text) and text.isupper():
        for old, new in replacements.items():
            text = text.replace(old, new)

    return text


def validate_id_card_fields(extracted_data: dict) -> dict:
    """
    Validate and clean extracted ID card fields.

    Args:
        extracted_data: Dictionary with extracted fields

    Returns:
        Validated and cleaned data
    """
    validated = extracted_data.copy()

    # Validate ID number
    if 'id_number' in validated and validated['id_number']:
        id_num = re.sub(r'[^\d]', '', validated['id_number'])
        if len(id_num) not in [9, 12]:
            validated['id_number'] = None
        else:
            validated['id_number'] = id_num

    # Validate dates
    date_fields = ['date_of_birth', 'date_of_expiry']
    for field in date_fields:
        if field in validated and validated[field]:
            date_text = validated[field]
            # Ensure DD/MM/YYYY format
            date_match = re.match(
                r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', date_text)
            if date_match:
                day, month, year = date_match.groups()
                validated[field] = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
            else:
                validated[field] = None

    # Validate gender
    if 'sex' in validated and validated['sex']:
        gender = validated['sex'].lower()
        if 'nam' in gender or 'male' in gender:
            validated['sex'] = 'Nam'
        elif 'nữ' in gender or 'female' in gender:
            validated['sex'] = 'Nữ'
        else:
            validated['sex'] = None

    # Clean text fields
    text_fields = ['full_name', 'place_of_origin',
                   'place_of_residence', 'nationality']
    for field in text_fields:
        if field in validated and validated[field]:
            validated[field] = normalize_vietnamese_text(validated[field])

    return validated
