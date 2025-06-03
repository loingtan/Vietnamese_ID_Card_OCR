#!/usr/bin/env python
"""
Test MongoDB connection.
"""

import sys

print(f"Python version: {sys.version}")
print("Testing MongoDB connection...")

try:
    from src.database.mongodb_fixed import MongoDBClient
    client = MongoDBClient()

    try:
        client.connect()
        print("MongoDB connection successful!")

        # Print database configuration
        print(f"MongoDB URL: {client.config.MONGODB_URL}")
        print(f"MongoDB Database: {client.config.MONGODB_DATABASE}")

        # Test count function
        count = client.get_ocr_results_count()
        print(f"Total OCR results: {count}")

    except Exception as e:
        print(f"MongoDB connection failed: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you are in the correct directory and have installed the required packages")
