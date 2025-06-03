#!/usr/bin/env python3
"""
Debug script để test Slack webhook trực tiếp
"""

import requests
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_slack_webhook_direct():
    """Test Slack webhook trực tiếp"""

    # Lấy URL từ environment hoặc sử dụng default
    slack_webhook_url = os.getenv(
        "SLACK_WEBHOOK_URL",
        "https://hooks.slack.com/services/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"
    )

    print(f"Testing Slack webhook URL: {slack_webhook_url}")

    # Test message
    test_message = {
        "text": "🧪 Test từ Vietnamese ID Card OCR Debug Script",
        "attachments": [
            {
                "color": "good",
                "text": f"Đây là test message được gửi lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "fields": [
                    {
                        "title": "Status",
                        "value": "Testing Slack Integration",
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": datetime.now().isoformat(),
                        "short": True
                    }
                ]
            }
        ]
    }

    try:
        print("Sending test message to Slack...")
        response = requests.post(
            slack_webhook_url,
            json=test_message,
            timeout=10
        )

        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")

        if response.status_code == 200:
            print("✅ Slack webhook hoạt động tốt!")
            return True
        else:
            print(
                f"❌ Slack webhook failed với status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_alert_processor():
    """Test AlertProcessor class trực tiếp"""

    try:
        # Import AlertProcessor
        import sys
        sys.path.append('src')
        from src.webhooks.alert_handlers import AlertProcessor, Alert, AlertStatus

        print("Testing AlertProcessor...")

        # Create AlertProcessor instance
        processor = AlertProcessor()
        print(
            f"Slack webhook URL configured: {bool(processor.slack_webhook_url)}")
        print(
            f"Telegram configured: {bool(processor.telegram_bot_token and processor.telegram_chat_id)}")

        # Create test alert
        test_alert = Alert(
            status=AlertStatus.FIRING,
            labels={
                "alertname": "TestAlert",
                "severity": "warning",
                "instance": "test-instance:8000"
            },
            annotations={
                "description": "Test alert from debug script",
                "summary": "Debug test alert"
            },
            startsAt=datetime.utcnow().isoformat() + "Z",
            generatorURL="http://localhost:9090/test",
            fingerprint="debug-test-123"
        )

        # Format message
        from src.webhooks.alert_handlers import AlertmanagerWebhook
        test_webhook = AlertmanagerWebhook(
            receiver="debug-test",
            status=AlertStatus.FIRING,
            alerts=[test_alert],
            groupLabels={"alertname": "TestAlert"},
            commonLabels=test_alert.labels,
            commonAnnotations=test_alert.annotations,
            externalURL="http://localhost:9093",
            version="1.0",
            groupKey="debug-group"
        )

        message = processor._format_alert_message(test_alert, test_webhook)
        print(f"Formatted message:\n{message}")

        # Test sending to Slack
        print("\nTesting Slack send...")
        processor._send_to_slack(message)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing AlertProcessor: {e}")
        return False


def check_environment():
    """Kiểm tra environment variables"""
    print("Checking environment variables...")

    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

    print(f"SLACK_WEBHOOK_URL: {'Set' if slack_url else 'Not set'}")
    print(f"TELEGRAM_BOT_TOKEN: {'Set' if telegram_token else 'Not set'}")
    print(f"TELEGRAM_CHAT_ID: {'Set' if telegram_chat else 'Not set'}")

    if slack_url:
        print(f"Slack URL starts with: {slack_url[:30]}...")


def main():
    """Main test function"""
    print("=== Vietnamese ID Card OCR - Slack Webhook Debug ===\n")

    # Check environment
    check_environment()
    print()

    # Test 1: Direct webhook test
    print("1. Testing direct Slack webhook...")
    if test_slack_webhook_direct():
        print("✅ Direct webhook test passed\n")
    else:
        print("❌ Direct webhook test failed\n")

    # Test 2: AlertProcessor test
    print("2. Testing AlertProcessor...")
    if test_alert_processor():
        print("✅ AlertProcessor test passed\n")
    else:
        print("❌ AlertProcessor test failed\n")

    print("=== Debug completed ===")


if __name__ == "__main__":
    main()
