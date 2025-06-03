#!/usr/bin/env python3
"""
Test script to verify Unicode encoding fixes for Vietnamese text logging.
"""

import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test Vietnamese text samples
VIETNAMESE_TEST_TEXTS = [
    "Nguyễn Văn Anh",
    "Trần Thị Bích",
    "Lê Hoàng Minh",
    "Phạm Thị Hương",
    "Hoàng Văn Đức",
    "Vũ Thị Linh",
    "Đặng Minh Tuấn",
    "Bùi Thị Nga",
    "Dương Văn Thắng",
    "Mai Thị Xuân",
    "Quảng Ninh",
    "Thành phố Hồ Chí Minh",
    "Hà Nội",
    "Đà Nẵng",
    "Nghệ An",
    "Thừa Thiên Huế"
]


def test_basic_unicode_logging():
    """Test basic Unicode logging functionality."""
    print("Testing basic Unicode logging...")

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/unicode_test.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('unicode_test')

    try:
        for text in VIETNAMESE_TEST_TEXTS:
            logger.info(f"Testing Vietnamese text: {text}")

        # Test JSON logging with Vietnamese text
        test_data = {
            "name": "Nguyễn Văn Anh",
            "id_number": "123456789",
            "place_of_birth": "Quảng Ninh",
            "address": "Thành phố Hồ Chí Minh",
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"JSON test: {json.dumps(test_data, ensure_ascii=False)}")
        print("✅ Basic Unicode logging test passed!")
        return True

    except UnicodeEncodeError as e:
        print(f"❌ Basic Unicode logging test failed: {e}")
        return False


def test_api_logging_setup():
    """Test the API logging setup with Vietnamese characters."""
    print("Testing API logging setup...")

    try:
        from src.api.fastapi_app import IDCardAPI

        # Create API instance (this will set up logging)
        api = IDCardAPI()

        # Test logging Vietnamese text through the API logger
        logger = logging.getLogger('src.api.fastapi_app')

        test_result = {
            "status": "success",
            "confidence": 0.95,
            "extracted_text": {
                "name": "Nguyễn Thị Hương",
                "id_number": "123456789",
                "place_of_birth": "Hà Nội",
                "address": "Quận Ba Đình, Hà Nội"
            }
        }

        # Test the _log_prediction method
        api._log_prediction(test_result, "test_vietnamese_id.jpg", 1.23)

        print("✅ API logging setup test passed!")
        return True

    except Exception as e:
        print(f"❌ API logging setup test failed: {e}")
        return False


def test_fallback_logging():
    """Test the fallback logging mechanism for Unicode errors."""
    print("Testing fallback logging mechanism...")

    try:
        # Simulate a logging scenario that might cause Unicode issues
        from src.api.fastapi_app import IDCardAPI

        api = IDCardAPI()

        # Test with problematic Unicode characters
        test_result = {
            "status": "success",
            "confidence": 0.88,
            "extracted_text": {
                "name": "Nguyễn Văn Đức",  # Vietnamese with diacritics
                "special_chars": "àáâãèéêìíîïòóôõöùúûüýỳỹỷỵ"
            }
        }

        # This should work with our UTF-8 setup
        api._log_prediction(test_result, "vietnamese_test.jpg", 0.85)

        print("✅ Fallback logging test passed!")
        return True

    except Exception as e:
        print(f"❌ Fallback logging test failed: {e}")
        return False


def test_log_file_content():
    """Verify that log files contain Vietnamese text correctly."""
    print("Testing log file content...")

    try:
        # Check if logs directory exists and has files
        logs_dir = Path('logs')
        if not logs_dir.exists():
            print("❌ Logs directory doesn't exist")
            return False

        # Check for log files
        log_files = list(logs_dir.glob('*.log'))
        if not log_files:
            print("❌ No log files found")
            return False

        # Read and verify Vietnamese content in log files
        vietnamese_found = False
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check first 5
                    for vietnamese_text in VIETNAMESE_TEST_TEXTS[:5]:
                        if vietnamese_text in content:
                            vietnamese_found = True
                            print(
                                f"✅ Found Vietnamese text '{vietnamese_text}' in {log_file.name}")
                            break
            except UnicodeDecodeError as e:
                print(f"❌ Unicode decode error reading {log_file.name}: {e}")
                return False

        if vietnamese_found:
            print("✅ Log file content test passed!")
            return True
        else:
            print("❌ No Vietnamese text found in log files")
            return False

    except Exception as e:
        print(f"❌ Log file content test failed: {e}")
        return False


def main():
    """Run all Unicode encoding tests."""
    print("🧪 Starting Unicode encoding tests for Vietnamese ID Card OCR...")
    print("=" * 60)

    tests = [
        test_basic_unicode_logging,
        test_api_logging_setup,
        test_fallback_logging,
        test_log_file_content
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Unicode encoding tests passed! Vietnamese text logging should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
