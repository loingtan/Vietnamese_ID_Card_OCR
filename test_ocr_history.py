#!/usr/bin/env python
"""
Test script to interact with the OCR history endpoints.
"""

import requests
import json
import sys
from datetime import datetime
import uuid

# Base URL for the FastAPI server
BASE_URL = "http://localhost:8080"


def print_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def get_ocr_history():
    """Get all OCR history"""
    try:
        url = f"{BASE_URL}/ocr-history"
        print(f"Fetching OCR history from {url}...")

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {data['count']} records.")
            print_json(data)
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None


def get_ocr_history_by_session(session_id):
    """Get OCR history for a specific session"""
    try:
        url = f"{BASE_URL}/ocr-history/session/{session_id}"
        print(f"Fetching OCR history for session {session_id} from {url}...")

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(
                f"Success! Found {len(data['results'])} records for session.")
            print_json(data)
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None


def create_test_data():
    """Create test data in MongoDB"""
    try:
        # Create a test image file
        with open("test_image.jpg", "wb") as f:
            f.write(b"This is a test image")

        session_id = str(uuid.uuid4())
        print(f"Creating test data with session_id: {session_id}")

        # Upload the test file to the API
        url = f"{BASE_URL}/process-id-card/"
        files = {'file': ('test_image.jpg', open(
            'test_image.jpg', 'rb'), 'image/jpeg')}

        response = requests.post(url, files=files)
        if response.status_code == 200:
            data = response.json()
            print("Success! Test data created.")
            print_json(data)
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error creating test data: {e}")
        return None


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(
            "Usage: python test_ocr_history.py [get_all|get_session|create_test]")
        return

    command = sys.argv[1]

    if command == "get_all":
        get_ocr_history()
    elif command == "get_session":
        if len(sys.argv) < 3:
            print("Usage: python test_ocr_history.py get_session [session_id]")
            return
        session_id = sys.argv[2]
        get_ocr_history_by_session(session_id)
    elif command == "create_test":
        create_test_data()
    else:
        print(f"Unknown command: {command}")
        print(
            "Usage: python test_ocr_history.py [get_all|get_session|create_test]")


if __name__ == "__main__":
    main()
