#!/usr/bin/env python
"""
Test saving OCR results to MongoDB.
"""

import sys
import time
from datetime import datetime
from src.database.mongodb import MongoDBClient
from src.database.models import OCRResult

print(f"Python version: {sys.version}")
print("Testing OCR result saving to MongoDB...")

try:
    # Create MongoDB client
    client = MongoDBClient()
    client.connect()
    print("MongoDB connection successful!")

    # Create a test OCR result
    ocr_result = OCRResult(
        session_id="test_session",
        image_filename="test_image.jpg",
        extracted_info={
            "ID_number": "123456789012",
            "Name": "Nguyễn Văn Test",
            "Date_of_birth": "01/01/1990",
            "Gender": "Nam",
            "Nationality": "Việt Nam",
            "Place_of_origin": "Hà Nội",
            "Place_of_residence": "Hồ Chí Minh"
        },
        processing_time=1.234,
        success=True,
        error_message=None,
        timestamp=datetime.utcnow()
    )

    # Save the OCR result
    result_id = client.save_ocr_result(ocr_result)
    print(f"Saved OCR result with ID: {result_id}")

    # Verify it was saved
    count = client.get_ocr_results_count()
    print(f"Total OCR results: {count}")

    # Retrieve the result
    results = client.get_ocr_results_by_session("test_session")
    print(f"Retrieved {len(results)} OCR results for test session")

    # Print the first result
    if results:
        print("First result data:")
        for key, value in results[0].items():
            if key != "extracted_info":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {len(value)} extracted fields")
                for field_key, field_value in value.items():
                    print(f"    {field_key}: {field_value}")

except Exception as e:
    print(f"Error: {e}")
