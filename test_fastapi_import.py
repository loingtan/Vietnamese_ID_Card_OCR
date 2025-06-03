#!/usr/bin/env python3
"""
Test script to mimic FastAPI import behavior
"""
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_fastapi_import():
    print("=== Test FastAPI Import Behavior ===")

    # Mimic the import pattern from FastAPI app
    print("1. Testing relative import pattern...")
    try:
        # This would fail in standalone script, so we test the fallback
        from src.webhooks.alert_handlers import router as alert_router, alert_processor

        print(f"Successfully imported alert_router and alert_processor")
        print(
            f"Alert processor slack_webhook_url: {alert_processor.slack_webhook_url}")
        print(
            f"Bool(slack_webhook_url): {bool(alert_processor.slack_webhook_url)}")

        # Test the health endpoint function directly
        from src.webhooks.alert_handlers import webhook_health
        import asyncio

        # Run the health check
        health_result = asyncio.run(webhook_health())
        print(f"\nHealth endpoint result: {health_result}")

    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fastapi_import()
