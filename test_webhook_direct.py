#!/usr/bin/env python3
"""
Test đơn giản để kiểm tra vì sao webhook không hoạt động
"""

import sys
import os
sys.path.append('src')


def test_import_alert_handlers():
    """Test import alert handlers"""
    try:
        print("Testing import alert_handlers...")
        from src.webhooks.alert_handlers import router, AlertProcessor
        print("✅ Import thành công")

        # Test AlertProcessor
        processor = AlertProcessor()
        print(f"Slack URL configured: {bool(processor.slack_webhook_url)}")

        # Test message format
        from src.webhooks.alert_handlers import Alert, AlertStatus, AlertmanagerWebhook
        import datetime

        test_alert = Alert(
            status=AlertStatus.FIRING,
            labels={"alertname": "Test", "severity": "info"},
            annotations={"description": "Test alert"},
            startsAt=datetime.datetime.utcnow().isoformat() + "Z",
            generatorURL="http://test",
            fingerprint="test"
        )

        test_webhook = AlertmanagerWebhook(
            receiver="test",
            status=AlertStatus.FIRING,
            alerts=[test_alert],
            groupLabels={},
            commonLabels={},
            commonAnnotations={},
            externalURL="http://test",
            version="1.0",
            groupKey="test"
        )

        message = processor._format_alert_message(test_alert, test_webhook)
        print(f"Message formatted: {len(message)} chars")

        # Test Slack send
        print("Testing Slack send...")
        processor._send_to_slack(message)
        print("✅ Slack send completed")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webhook_routes():
    """Test webhook routes trực tiếp"""
    try:
        from fastapi import FastAPI
        from src.webhooks.alert_handlers import router

        app = FastAPI()
        app.include_router(router)

        # Get routes
        routes = [route.path for route in app.routes]
        print("Available routes:")
        for route in routes:
            print(f"  - {route}")

        webhook_routes = [r for r in routes if 'webhook' in r]
        print(f"Webhook routes found: {len(webhook_routes)}")

        return len(webhook_routes) > 0

    except Exception as e:
        print(f"❌ Error testing routes: {e}")
        return False


if __name__ == "__main__":
    print("=== Debug Webhook Issues ===\n")

    print("1. Testing alert handler imports...")
    if test_import_alert_handlers():
        print("✅ Alert handlers working\n")
    else:
        print("❌ Alert handlers failed\n")

    print("2. Testing webhook routes...")
    if test_webhook_routes():
        print("✅ Webhook routes working\n")
    else:
        print("❌ Webhook routes failed\n")
