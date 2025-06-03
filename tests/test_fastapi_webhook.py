#!/usr/bin/env python3
"""
Test FastAPI app với webhook
"""

from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
import os
sys.path.append('src')


def test_fastapi_webhook():
    """Test FastAPI webhook endpoint"""

    try:
        # Import và setup
        from src.api.fastapi_app import create_app

        app = create_app()
        client = TestClient(app)

        # Test health endpoint
        print("Testing /webhooks/health...")
        response = client.get("/webhooks/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test alert endpoint
        print("\nTesting /webhooks/test-alert...")
        response = client.post(
            "/webhooks/test-alert?alert_type=test&severity=info")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing FastAPI Webhook Integration ===")
    test_fastapi_webhook()
