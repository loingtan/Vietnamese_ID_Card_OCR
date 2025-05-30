"""
Vietnamese ID Card OCR - Alert Webhook Handlers
This module contains webhook endpoints for handling alerts from Alertmanager
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import json
import datetime
from enum import Enum
import requests
import os

# Try to import config, fallback to a simple config if not available
try:
    from ..config import get_config
    config = get_config()
except ImportError:
    # Fallback for testing
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.WEBHOOK_ENABLED = False
    config.SLACK_WEBHOOK_URL = ""
    config.DISCORD_WEBHOOK_URL = ""

# Configure logger
logger = logging.getLogger("alert_handler")


class AlertStatus(str, Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


class Alert(BaseModel):
    status: AlertStatus
    labels: Dict[str, str]
    annotations: Dict[str, str]
    startsAt: str
    endsAt: Optional[str] = None
    generatorURL: str
    fingerprint: str


class AlertmanagerWebhook(BaseModel):
    receiver: str
    status: AlertStatus
    alerts: List[Alert]
    groupLabels: Dict[str, str]
    commonLabels: Dict[str, str]
    commonAnnotations: Dict[str, str]
    externalURL: str
    version: str
    groupKey: str
    truncatedAlerts: int = 0


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class NotificationChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"


# Create router
router = APIRouter(prefix="/webhooks", tags=["alerts"])


class AlertProcessor:
    """Process and route alerts to appropriate notification channels"""

    def __init__(self):
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def process_alert(self, webhook_data: AlertmanagerWebhook) -> None:
        """Process incoming alert webhook"""
        logger.info(
            f"Processing {len(webhook_data.alerts)} alerts from {webhook_data.receiver}")

        for alert in webhook_data.alerts:
            self._handle_single_alert(alert, webhook_data)

    def _handle_single_alert(self, alert: Alert, webhook_data: AlertmanagerWebhook) -> None:
        """Handle a single alert"""
        severity = self._get_alert_severity(alert)
        alert_message = self._format_alert_message(alert, webhook_data)

        logger.info(
            f"Alert: {alert.labels.get('alertname', 'Unknown')} - {severity} - {alert.status}")

        # Route to appropriate notification channels based on severity and type
        if severity == AlertSeverity.CRITICAL:
            self._send_to_all_channels(alert_message, alert)
        elif severity == AlertSeverity.WARNING:
            self._send_to_selected_channels(
                alert_message, alert, [NotificationChannel.SLACK, NotificationChannel.EMAIL])
        else:  # INFO
            self._send_to_selected_channels(
                alert_message, alert, [NotificationChannel.SLACK])

    def _get_alert_severity(self, alert: Alert) -> AlertSeverity:
        """Determine alert severity"""
        severity = alert.labels.get("severity", "info").lower()

        if severity in ["critical", "fatal"]:
            return AlertSeverity.CRITICAL
        elif severity in ["warning", "warn"]:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _format_alert_message(self, alert: Alert, webhook_data: AlertmanagerWebhook) -> str:
        """Format alert message for notifications"""
        alertname = alert.labels.get("alertname", "Unknown Alert")
        instance = alert.labels.get("instance", "Unknown Instance")
        severity = self._get_alert_severity(alert)

        # Get description from annotations
        description = alert.annotations.get(
            "description", alert.annotations.get("summary", "No description available"))

        # Format timestamp
        starts_at = datetime.datetime.fromisoformat(
            alert.startsAt.replace('Z', '+00:00'))

        # Build message
        status_emoji = "🔥" if alert.status == AlertStatus.FIRING else "✅"
        severity_emoji = {"critical": "🚨",
                          "warning": "⚠️", "info": "ℹ️"}[severity]

        message = f"""
{status_emoji} {severity_emoji} **{alertname}** - {alert.status.upper()}

**Instance:** {instance}
**Severity:** {severity.upper()}
**Started:** {starts_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

**Description:**
{description}

**Labels:**
{self._format_labels(alert.labels)}

**Generator URL:** {alert.generatorURL}
        """.strip()

        return message

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for display"""
        formatted = []
        for key, value in labels.items():
            if key not in ["__name__", "alertname"]:  # Skip internal labels
                formatted.append(f"  • {key}: {value}")
        return "\n".join(formatted) if formatted else "  None"

    def _send_to_all_channels(self, message: str, alert: Alert) -> None:
        """Send critical alerts to all notification channels"""
        self._send_to_slack(message)
        self._send_to_telegram(message)
        self._send_to_email(message, alert)

    def _send_to_selected_channels(self, message: str, alert: Alert, channels: List[NotificationChannel]) -> None:
        """Send alerts to selected notification channels"""
        if NotificationChannel.SLACK in channels:
            self._send_to_slack(message)
        if NotificationChannel.TELEGRAM in channels:
            self._send_to_telegram(message)
        if NotificationChannel.EMAIL in channels:
            self._send_to_email(message, alert)

    def _send_to_slack(self, message: str) -> None:
        """Send notification to Slack"""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        try:
            # Convert markdown to Slack format
            slack_message = message.replace("**", "*")

            payload = {
                "text": "Vietnamese ID Card OCR Alert",
                "attachments": [
                    {
                        "text": slack_message,
                        "color": "danger" if "🔥" in message else "warning" if "⚠️" in message else "good"
                    }
                ]
            }

            response = requests.post(
                self.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Alert sent to Slack successfully")

        except Exception as e:
            logger.error(f"Failed to send alert to Slack: {e}")

    def _send_to_telegram(self, message: str) -> None:
        """Send notification to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Alert sent to Telegram successfully")

        except Exception as e:
            logger.error(f"Failed to send alert to Telegram: {e}")

    def _send_to_email(self, message: str, alert: Alert) -> None:
        """Send notification via email (placeholder - implement with your email service)"""
        # This is a placeholder - implement with your preferred email service
        # (SMTP, SendGrid, AWS SES, etc.)
        logger.info(
            f"Email notification would be sent for alert: {alert.labels.get('alertname')}")


# Initialize alert processor
alert_processor = AlertProcessor()


@router.post("/alertmanager")
async def alertmanager_webhook(
    webhook_data: AlertmanagerWebhook,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Webhook endpoint for Alertmanager notifications

    This endpoint receives alerts from Alertmanager and processes them
    according to their severity and type.
    """
    try:
        # Log the incoming webhook
        logger.info(
            f"Received webhook from Alertmanager: {webhook_data.receiver}")
        logger.debug(f"Webhook data: {webhook_data.dict()}")

        # Process alerts in background to avoid blocking the response
        background_tasks.add_task(alert_processor.process_alert, webhook_data)

        return {
            "status": "success",
            "message": f"Processed {len(webhook_data.alerts)} alerts",
            "receiver": webhook_data.receiver,
            "group_key": webhook_data.groupKey
        }

    except Exception as e:
        logger.error(f"Error processing alertmanager webhook: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing webhook: {str(e)}")


@router.post("/test-alert")
async def test_alert_webhook(
    alert_type: str = "test",
    severity: AlertSeverity = AlertSeverity.INFO,
    background_tasks: BackgroundTasks = None
):
    """
    Test endpoint for alert notifications

    Use this endpoint to test your notification channels
    """
    try:
        # Create a test alert
        test_alert = Alert(
            status=AlertStatus.FIRING,
            labels={
                "alertname": f"TestAlert{alert_type.capitalize()}",
                "severity": severity.value,
                "instance": "test-instance:8000",
                "job": "vietnamese-id-ocr-api",
                "service": "api"
            },
            annotations={
                "description": f"This is a test {severity.value} alert for the Vietnamese ID Card OCR system",
                "summary": f"Test alert - {alert_type}"
            },
            startsAt=datetime.datetime.utcnow().isoformat() + "Z",
            generatorURL="http://localhost:9090/test",
            fingerprint="test-fingerprint-123"
        )

        # Create test webhook data
        test_webhook = AlertmanagerWebhook(
            receiver="webhook-test",
            status=AlertStatus.FIRING,
            alerts=[test_alert],
            groupLabels={"alertname": test_alert.labels["alertname"]},
            commonLabels=test_alert.labels,
            commonAnnotations=test_alert.annotations,
            externalURL="http://localhost:9093",
            version="1.0",
            groupKey="test-group"
        )

        # Process the test alert
        if background_tasks:
            background_tasks.add_task(
                alert_processor.process_alert, test_webhook)
        else:
            alert_processor.process_alert(test_webhook)

        return {
            "status": "success",
            "message": f"Test {severity.value} alert sent",
            "alert_type": alert_type,
            "severity": severity.value
        }

    except Exception as e:
        logger.error(f"Error sending test alert: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error sending test alert: {str(e)}")


@router.get("/health")
async def webhook_health():
    """Health check endpoint for webhook service"""
    return {
        "status": "healthy",
        "service": "alert-webhooks",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "configured_channels": {
            "slack": bool(alert_processor.slack_webhook_url),
            "telegram": bool(alert_processor.telegram_bot_token and alert_processor.telegram_chat_id),
            "email": False  # Update when email is implemented
        }
    }
