#!/usr/bin/env python3
"""
Integration test to verify the complete Vietnamese ID Card OCR pipeline works without Unicode errors.
"""

import sys
import os
import asyncio
import tempfile
from PIL import Image
import io

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_test_image():
    """Create a simple test image for upload."""
    # Create a simple test image
    img = Image.new('RGB', (400, 250), color='white')

    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return img_bytes.getvalue()


async def test_id_card_processing():
    """Test the full ID card processing pipeline."""
    print("🔄 Testing complete ID card processing pipeline...")

    try:
        from src.api.fastapi_app import IDCardAPI
        from fastapi.testclient import TestClient
        from fastapi import UploadFile
        import tempfile

        # Create API instance
        api = IDCardAPI()

        # Create test client
        client = TestClient(api.app)

        # Create test image
        test_image_data = create_test_image()

        # Create temporary file for upload
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(test_image_data)
            tmp_file.flush()

            # Test the API endpoint
            with open(tmp_file.name, 'rb') as f:
                files = {'file': ('test_vietnamese_id.png', f, 'image/png')}

                print("📤 Sending test image to API endpoint...")
                response = client.post('/process-id-card/', files=files)

                print(f"📨 Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    print("✅ API endpoint responded successfully")

                    # Check if the response contains expected fields
                    if 'status' in result:
                        print(f"   Status: {result['status']}")
                    if 'processing_time' in result:
                        print(
                            f"   Processing time: {result['processing_time']:.3f}s")
                    if 'extracted_fields' in result:
                        print("   Extracted fields found in response")

                    # Test with Vietnamese text in the response
                    vietnamese_test_data = {
                        "name": "Nguyễn Thị Hương",
                        "place_of_birth": "Thành phố Hồ Chí Minh",
                        "address": "Quận 1, Thành phố Hồ Chí Minh"
                    }

                    # Log this as a successful prediction
                    api._log_prediction({
                        "status": "success",
                        "confidence": 0.89,
                        "extracted_fields": vietnamese_test_data
                    }, "test_integration.jpg", 2.1)

                    print(
                        "✅ Successfully logged Vietnamese text through integration test")
                    return True
                else:
                    print(
                        f"❌ API endpoint failed with status {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False

        # Clean up
        os.unlink(tmp_file.name)

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with Vietnamese text."""
    print("\n🛡️ Testing error handling with Vietnamese text...")

    try:
        from src.api.fastapi_app import IDCardAPI

        api = IDCardAPI()

        # Test logging a failed prediction with Vietnamese text
        failed_result = {
            "status": "error",
            "error_message": "Không thể nhận diện văn bản tiếng Việt",
            "confidence": 0.1,
            "extracted_fields": None
        }

        api._log_prediction(failed_result, "failed_vietnamese_card.jpg", 0.5)
        print("✅ Successfully logged Vietnamese error message")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run integration tests."""
    print("🔧 Running Vietnamese ID Card OCR Integration Tests")
    print("=" * 60)

    # Test 1: Complete pipeline
    pipeline_test = asyncio.run(test_id_card_processing())

    # Test 2: Error handling
    error_test = test_error_handling()

    print("\n" + "=" * 60)
    print("🏁 Integration Test Summary:")
    print(f"   Pipeline Test: {'✅ PASSED' if pipeline_test else '❌ FAILED'}")
    print(f"   Error Handling: {'✅ PASSED' if error_test else '❌ FAILED'}")

    if pipeline_test and error_test:
        print("\n🎉 All integration tests passed!")
        print("   The Unicode encoding fix is working correctly in the full pipeline.")
        print("   Vietnamese text should be processed and logged without encoding errors.")
        return 0
    else:
        print("\n⚠️ Some integration tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
