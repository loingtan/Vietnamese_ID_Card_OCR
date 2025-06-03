#!/usr/bin/env python3
"""
Test simple để kiểm tra webhook Slack
"""

import requests
import logging
import os

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_slack_webhook():
    """Test Slack webhook đơn giản"""

    # URL từ code
    slack_webhook_url = "https://hooks.slack.com/services/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"

    message = """🔥 ⚠️ **TestAlert** - FIRING

**Instance:** test-instance
**Severity:** WARNING
**Started:** 2025-06-03 10:30:00 UTC

**Description:**
This is a test alert

**Labels:**
  • service: vnidcard-api
  • severity: warning

**Generator URL:** http://localhost:9090/test"""

    try:
        # Convert markdown to Slack format
        slack_message = message.replace("**", "*")

        payload = {
            "text": "Vietnamese ID Card OCR Alert",
            "attachments": [
                {
                    "text": slack_message,
                    "color": "warning"
                }
            ]
        }

        print("Sending to Slack...")
        print(f"URL: {slack_webhook_url[:50]}...")
        print(f"Payload: {payload}")

        response = requests.post(slack_webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        print("✅ Alert sent to Slack successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to send alert to Slack: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing Slack Webhook Directly ===")
    test_slack_webhook()
