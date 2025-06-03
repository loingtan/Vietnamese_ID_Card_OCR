import requests
import json
import argparse
from datetime import datetime


def send_test_alert(alert_type="default", service_name="test-service"):
    """
    Send test alert to Slack webhook URL configured in AlertManager

    Args:
        alert_type: Type of alert (default, critical, api, system, gpu, model)
        service_name: Name of the service generating the alert
    """
    # This is the Slack webhook URL from your AlertManager config
    webhook_url = "https://hooks.slack.com/services/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"

    current_time = datetime.now().isoformat()

    # Prepare alert content based on the alert type
    if alert_type == "critical":
        message = {
            "attachments": [
                {
                    "color": "danger",
                    "title": "🚨 CRITICAL: Vietnamese ID Card API",
                    "text": f"🚨 **CRITICAL ALERT**\n**Service:** {service_name}\n**Alert:** Test Critical Alert\n**Description:** This is a test critical alert for {service_name}\n**Time:** {current_time}\n\n@here Please investigate immediately!"
                }
            ]
        }
    elif alert_type == "api":
        message = {
            "attachments": [
                {
                    "color": "warning",
                    "title": "⚠️ API Alert: Vietnamese ID Card",
                    "text": f"**Service:** {service_name}\n**Alert:** Test API Alert\n**Description:** This is a test API alert for {service_name}"
                }
            ]
        }
    elif alert_type == "system":
        message = {
            "attachments": [
                {
                    "color": "warning",
                    "title": "🖥️ System Alert: Vietnamese ID Card",
                    "text": f"**Component:** {service_name}\n**Alert:** Test System Alert\n**Description:** This is a test system alert\n**Instance:** test-instance"
                }
            ]
        }
    elif alert_type == "gpu":
        message = {
            "attachments": [
                {
                    "color": "warning",
                    "title": "🎮 GPU Alert: Vietnamese ID Card ML",
                    "text": f"**GPU:** test-gpu\n**Alert:** Test GPU Alert\n**Description:** This is a test GPU alert for {service_name}"
                }
            ]
        }
    elif alert_type == "model":
        message = {
            "attachments": [
                {
                    "color": "warning",
                    "title": "🤖 Model Alert: Vietnamese ID Card ML",
                    "text": f"**Model:** test-model\n**Alert:** Test Model Alert\n**Description:** This is a test model performance alert for {service_name}"
                }
            ]
        }
    else:  # default alert
        message = {
            "attachments": [
                {
                    "color": "warning",
                    "title": "Vietnamese ID Card API Alert",
                    "text": f"**Alert:** Test Default Alert\n**Service:** {service_name}\n**Description:** This is a default test alert"
                }
            ]
        }

    print(f"Sending {alert_type} test alert for service: {service_name}")

    # Send the message to Slack
    response = requests.post(
        webhook_url,
        data=json.dumps(message),
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code == 200:
        print("Test alert sent successfully!")
    else:
        print(
            f"Failed to send test alert. Status code: {response.status_code}")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Slack webhook alerts for AlertManager")
    parser.add_argument(
        "--type",
        choices=["default", "critical", "api", "system", "gpu", "model"],
        default="default",
        help="Type of alert to test"
    )
    parser.add_argument(
        "--service",
        default="vnidcard-api",
        help="Name of the service generating the alert"
    )

    args = parser.parse_args()
    send_test_alert(args.type, args.service)
