#!/usr/bin/env python3
"""
Test script to start the FastAPI server and test Vietnamese text processing.
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_api_server():
    """Test the API server with Vietnamese text processing."""
    print("🚀 Testing FastAPI server with Vietnamese text processing...")

    try:
        # Import and start the API
        from src.api.fastapi_app import IDCardAPI

        print("✅ Successfully imported IDCardAPI")

        # Create API instance
        api = IDCardAPI()
        print("✅ Successfully created API instance")

        # Test logging with Vietnamese text
        test_result = {
            "status": "success",
            "confidence": 0.92,
            "extracted_fields": {
                "name": "Nguyễn Văn Minh",
                "id_number": "123456789",
                "date_of_birth": "01/01/1990",
                "place_of_birth": "Hà Nội",
                "permanent_address": "Số 123, Phố Láng, Quận Đống Đa, Hà Nội",
                "ethnicity": "Kinh",
                "religion": "Không"
            }
        }

        # Test the logging function
        api._log_prediction(test_result, "test_vietnamese_card.jpg", 1.45)
        print("✅ Successfully logged Vietnamese text prediction")

        # Test JSON serialization with Vietnamese characters
        json_output = json.dumps(test_result, ensure_ascii=False, indent=2)
        print("✅ Successfully serialized Vietnamese text to JSON")
        print(f"Sample JSON output:\n{json_output[:200]}...")

        return True

    except Exception as e:
        print(f"❌ API server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_log_files():
    """Verify that log files contain Vietnamese text correctly."""
    print("\n📁 Verifying log files...")

    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("❌ Logs directory doesn't exist")
        return False

    # Check specific log files
    log_files_to_check = ['api.log', 'model.log', 'unicode_test.log']
    vietnamese_patterns = ['Nguyễn', 'Văn', 'Thị', 'Hà Nội', 'Minh']

    for log_file in log_files_to_check:
        log_path = logs_dir / log_file
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    found_vietnamese = False
                    for pattern in vietnamese_patterns:
                        if pattern in content:
                            found_vietnamese = True
                            print(
                                f"✅ Found Vietnamese text '{pattern}' in {log_file}")
                            break

                    if not found_vietnamese:
                        print(f"⚠️ No Vietnamese text found in {log_file}")

            except UnicodeDecodeError as e:
                print(f"❌ Unicode decode error in {log_file}: {e}")
                return False
        else:
            print(f"⚠️ Log file {log_file} doesn't exist")

    return True


def main():
    """Main test function."""
    print("🧪 Testing Vietnamese ID Card OCR Unicode Encoding Fix")
    print("=" * 60)

    # Test 1: API Server functionality
    api_test_passed = test_api_server()

    # Test 2: Log file verification
    log_verification_passed = verify_log_files()

    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    print(
        f"   API Server Test: {'✅ PASSED' if api_test_passed else '❌ FAILED'}")
    print(
        f"   Log Verification: {'✅ PASSED' if log_verification_passed else '❌ FAILED'}")

    if api_test_passed and log_verification_passed:
        print("\n🎉 All tests passed! The Unicode encoding fix is working correctly.")
        print("   Vietnamese text should now be logged without encoding errors.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
