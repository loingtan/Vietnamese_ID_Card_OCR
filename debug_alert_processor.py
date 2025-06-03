#!/usr/bin/env python3
"""
Debug script to check AlertProcessor initialization
"""
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def debug_alert_processor():
    print("=== Debug AlertProcessor Initialization ===")

    # Check environment variables
    print(f"Environment SLACK_WEBHOOK_URL: {os.getenv('SLACK_WEBHOOK_URL')}")

    # Try to import and initialize AlertProcessor
    try:
        from src.webhooks.alert_handlers import AlertProcessor, alert_processor

        print(f"\nAlertProcessor instance:")
        print(f"  slack_webhook_url: {alert_processor.slack_webhook_url}")
        print(f"  telegram_bot_token: {alert_processor.telegram_bot_token}")
        print(f"  telegram_chat_id: {alert_processor.telegram_chat_id}")

        # Test bool conversion
        print(f"\nBool checks:")
        print(
            f"  bool(slack_webhook_url): {bool(alert_processor.slack_webhook_url)}")
        print(
            f"  slack_webhook_url is None: {alert_processor.slack_webhook_url is None}")
        print(
            f"  slack_webhook_url == '': {alert_processor.slack_webhook_url == ''}")
        print(
            f"  len(slack_webhook_url): {len(alert_processor.slack_webhook_url) if alert_processor.slack_webhook_url else 'N/A'}")

        # Create a new instance to test initialization
        print(f"\nCreating new AlertProcessor instance:")
        new_processor = AlertProcessor()
        print(
            f"  New instance slack_webhook_url: {new_processor.slack_webhook_url}")
        print(
            f"  New instance bool(slack_webhook_url): {bool(new_processor.slack_webhook_url)}")

    except Exception as e:
        print(f"Error importing AlertProcessor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_alert_processor()
