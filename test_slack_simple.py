#!/usr/bin/env python3
"""
Simple test để kiểm tra vì sao Slack webhook không hoạt động
"""

import requests
import os


def test_slack_simple():
    """Test Slack webhook đơn giản"""

    # Thử các URL khác nhau
    urls_to_test = [
        "https://hooks.slack.com/services/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",
        "https://hooks.slack.com/services/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",
        os.getenv("SLACK_WEBHOOK_URL")
    ]

    test_payload = {
        "text": "🧪 Test message từ Vietnamese ID Card OCR",
        "username": "VN-ID-OCR-Bot",
        "icon_emoji": ":robot_face:"
    }

    for i, url in enumerate(urls_to_test):
        if not url:
            print(f"URL {i+1}: Không có URL")
            continue

        print(f"Testing URL {i+1}: {url[:50]}...")

        try:
            response = requests.post(url, json=test_payload, timeout=10)
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")

            if response.status_code == 200:
                print(f"  ✅ URL {i+1} hoạt động!")
                return True
            else:
                print(f"  ❌ URL {i+1} failed")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return False


if __name__ == "__main__":
    print("=== Testing Slack Webhooks ===")

    # Set environment variable for testing
    os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/services/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"

    test_slack_simple()
