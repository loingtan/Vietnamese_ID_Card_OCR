import requests
import json
import argparse
from datetime import datetime, timedelta


def send_test_alert_to_alertmanager(
    alert_type="default",
    service="vnidcard-api",
    severity="warning",
    alertmanager_url="http://localhost:9093"
):
    """
    Send test alert directly to AlertManager

    Args:
        alert_type: Type of alert (default, api, system, gpu, model)
        service: Name of the service generating the alert
        severity: Alert severity (warning, critical)
        alertmanager_url: URL of the AlertManager instance
    """
    current_time = datetime.now().isoformat()
    end_time = (datetime.now() + timedelta(minutes=5)).isoformat()

    # Set up alert details based on type
    if alert_type == "api":
        alert_name = "ApiLatencyHigh"
        summary = "API latency is high"
        description = "API response time is exceeding thresholds"
    elif alert_type == "system":
        alert_name = "HighCpuUsage"
        summary = "High CPU usage detected"
        description = "System CPU usage is above 90%"
        service = "system"  # Override service for system alerts
    elif alert_type == "gpu":
        alert_name = "GpuMemoryHigh"
        summary = "High GPU memory usage"
        description = "GPU memory utilization above threshold"
        service = "gpu"  # Override service for GPU alerts
    elif alert_type == "model":
        alert_name = "ModelAccuracyDrop"
        summary = "Model accuracy decreased"
        description = "ML model accuracy has dropped below acceptable threshold"
        service = "vnidcard-model"  # Override service for model alerts
    else:  # default
        alert_name = "GeneralAlert"
        summary = "General test alert"
        description = "This is a general test alert for AlertManager"

    # Create the alert payload
    alert = [{
        "status": "firing",
        "labels": {
            "alertname": alert_name,
            "service": service,
            "severity": severity,
            "instance": "test-instance:9090"
        },
        "annotations": {
            "summary": summary,
            "description": description
        },
        "startsAt": current_time,
        "endsAt": end_time,
        "generatorURL": "http://localhost:9090/graph"
    }]

    print(f"Sending {severity} {alert_type} alert for service: {service}")
    print(json.dumps(alert, indent=2))

    # Send the alert to AlertManager
    try:
        response = requests.post(
            f"{alertmanager_url}/api/v1/alerts",
            data=json.dumps(alert),
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            print("Alert sent successfully to AlertManager!")
        else:
            print(f"Failed to send alert. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to AlertManager: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test alerts for AlertManager")
    parser.add_argument(
        "--type",
        choices=["default", "api", "system", "gpu", "model"],
        default="default",
        help="Type of alert to test"
    )
    parser.add_argument(
        "--severity",
        choices=["warning", "critical"],
        default="warning",
        help="Severity of the alert"
    )
    parser.add_argument(
        "--service",
        default="vnidcard-api",
        help="Name of the service generating the alert (for API alerts)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9093",
        help="URL of the AlertManager instance"
    )

    args = parser.parse_args()
    send_test_alert_to_alertmanager(
        args.type,
        args.service,
        args.severity,
        args.url
    )
