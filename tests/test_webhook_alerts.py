"""
Test script for the AlertManager webhook alerts
This script sends test alerts to your API's webhook endpoint
"""

import requests
import argparse
import sys
import json


def send_test_alert(base_url="http://localhost:8000", alert_type="test", severity="info"):
    """
    Send a test alert to the webhook endpoint

    Args:
        base_url: The base URL of your API
        alert_type: Type of alert to test
        severity: Severity level (info, warning, critical)
    """
    endpoint = f"{base_url}/webhooks/test-alert"

    params = {
        "alert_type": alert_type,
        "severity": severity
    }

    print(f"Sending {severity} test alert '{alert_type}' to {endpoint}")

    try:
        response = requests.post(endpoint, params=params, timeout=10)

        # Print response details
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print("\n✅ Test alert sent successfully!")
        else:
            print(f"❌ Error: {response.text}")

    except requests.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("\nIs your API running at {base_url}?")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the alert webhook endpoint")

    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of your API (default: http://localhost:8000)")

    parser.add_argument("--type", default="test",
                        help="Type of alert to test (default: test)")

    parser.add_argument("--severity", choices=["info", "warning", "critical"],
                        default="info", help="Severity level (default: info)")

    args = parser.parse_args()

    send_test_alert(args.url, args.type, args.severity)
