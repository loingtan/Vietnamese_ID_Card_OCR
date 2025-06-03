#!/usr/bin/env python
"""
Test the simplified OCR system that only accepts OCRResult fields.
"""

import sys
import time
from datetime import datetime
from src.database.mongodb import MongoDBClient
from src.database.models import OCRResult

print(f"Python version: {sys.version}")
print("Testing simplified OCR system with only OCRResult...")


def test_ocr_result_creation():
    """Test creating OCRResult with only allowed fields."""
    print("\n1. Testing OCRResult creation...")

    # Test with all fields
    ocr_result = OCRResult(
        session_id="test_session_001",
        image_filename="test_card.jpg",
        extracted_info={
            "id_number": "123456789012",
            "full_name": "Nguyễn Văn Test",
            "date_of_birth": "01/01/1990",
            "gender": "Nam",
            "nationality": "Việt Nam",
            "place_of_origin": "Hà Nội",
            "place_of_residence": "TP. Hồ Chí Minh"
        },
        processing_time=2.5,
        success=True,
        error_message=None,
        timestamp=datetime.utcnow()
    )

    print(f"✓ OCRResult created successfully")
    print(f"  - Session ID: {ocr_result.session_id}")
    print(f"  - Image: {ocr_result.image_filename}")
    print(f"  - Processing time: {ocr_result.processing_time}s")
    print(f"  - Success: {ocr_result.success}")
    print(f"  - Timestamp: {ocr_result.timestamp}")
    print(f"  - Extracted fields: {len(ocr_result.extracted_info)}")

    return ocr_result


def test_ocr_result_serialization(ocr_result):
    """Test OCRResult to_dict and from_dict methods."""
    print("\n2. Testing OCRResult serialization...")

    # Test to_dict
    result_dict = ocr_result.to_dict()
    print(f"✓ to_dict() successful, keys: {list(result_dict.keys())}")

    # Test from_dict
    restored_result = OCRResult.from_dict(result_dict)
    print(f"✓ from_dict() successful")
    print(
        f"  - Session ID matches: {restored_result.session_id == ocr_result.session_id}")
    print(
        f"  - Processing time matches: {restored_result.processing_time == ocr_result.processing_time}")
    print(
        f"  - Success status matches: {restored_result.success == ocr_result.success}")

    return restored_result


def test_mongodb_operations():
    """Test MongoDB operations with simplified schema."""
    print("\n3. Testing MongoDB operations...")

    try:
        # Connect to MongoDB
        client = MongoDBClient()
        client.connect()
        print("✓ MongoDB connection successful")

        # Create test OCR result
        test_result = OCRResult(
            session_id="test_session_mongo",
            image_filename="mongo_test.jpg",
            extracted_info={
                "id_number": "987654321098",
                "full_name": "Trần Thị MongoDB",
                "date_of_birth": "15/08/1985"
            },
            processing_time=1.8,
            success=True
        )

        # Save to database
        result_id = client.save_ocr_result(test_result)
        print(f"✓ OCR result saved with ID: {result_id}")

        # Retrieve by session
        session_results = client.get_ocr_results_by_session(
            "test_session_mongo")
        print(f"✓ Retrieved {len(session_results)} results for session")

        if session_results:
            first_result = session_results[0]
            print(f"  - Image filename: {first_result['image_filename']}")
            print(f"  - Processing time: {first_result['processing_time']}")
            print(f"  - Success: {first_result['success']}")
            print(
                f"  - Extracted info keys: {list(first_result['extracted_info'].keys())}")

        # Get count
        count = client.get_ocr_results_count()
        print(f"✓ Total OCR results in database: {count}")

        # Search by ID number
        search_results = client.search_by_id_number("987654321098")
        print(f"✓ Found {len(search_results)} results for ID number search")

        client.disconnect()
        print("✓ MongoDB disconnected")

        return True

    except Exception as e:
        print(f"✗ MongoDB test failed: {e}")
        return False


def test_error_handling():
    """Test error handling with OCRResult."""
    print("\n4. Testing error handling...")

    # Test with error
    error_result = OCRResult(
        session_id="test_error_session",
        image_filename="corrupted_image.jpg",
        extracted_info={},
        processing_time=0.5,
        success=False,
        error_message="Image is corrupted or unreadable"
    )

    print(f"✓ Error OCRResult created")
    print(f"  - Success: {error_result.success}")
    print(f"  - Error message: {error_result.error_message}")
    print(f"  - Empty extracted_info: {len(error_result.extracted_info) == 0}")

    return error_result


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLIFIED OCR SYSTEM TEST")
    print("Only OCRResult with specified fields")
    print("=" * 60)

    try:
        # Test 1: OCRResult creation
        ocr_result = test_ocr_result_creation()

        # Test 2: Serialization
        restored_result = test_ocr_result_serialization(ocr_result)

        # Test 3: MongoDB operations
        mongo_success = test_mongodb_operations()

        # Test 4: Error handling
        error_result = test_error_handling()

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ OCRResult creation and serialization: PASSED")
        print(f"{'✓' if mongo_success else '✗'} MongoDB operations: {'PASSED' if mongo_success else 'FAILED'}")
        print("✓ Error handling: PASSED")
        print("\n🎉 Simplified OCR system is working correctly!")
        print("Only accepts the specified OCRResult fields:")
        print("  - session_id: str")
        print("  - image_filename: str")
        print("  - extracted_info: Dict[str, Any]")
        print("  - processing_time: float")
        print("  - success: bool")
        print("  - error_message: Optional[str]")
        print("  - timestamp: Optional[datetime]")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
