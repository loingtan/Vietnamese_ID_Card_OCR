# Alerting System Test Tools

This folder contains tools for testing the alerting system of the Vietnamese ID Card application.

## Tools Included

1. `test_slack_webhook.py` - Direct test of Slack webhook functionality
2. `test_alertmanager.py` - Test the full AlertManager pipeline

## Testing Slack Webhooks Directly

The `test_slack_webhook.py` script allows you to test the Slack webhook directly, bypassing AlertManager.

### Usage

```bash
python test_slack_webhook.py [--type {default,critical,api,system,gpu,model}] [--service SERVICE]
```

### Options

- `--type`: Type of alert to test
  - Choices: default, critical, api, system, gpu, model
  - Default: default
- `--service`: Name of the service generating the alert
  - Default: vnidcard-api

### Examples

```bash
# Test a default alert
python test_slack_webhook.py

# Test a critical alert
python test_slack_webhook.py --type critical

# Test an API alert for a specific service
python test_slack_webhook.py --type api --service "vnidcard-api-staging"
```

## Testing AlertManager Pipeline

The `test_alertmanager.py` script allows you to send test alerts directly to your AlertManager instance. This tests the complete alerting pipeline, including the routing rules defined in your AlertManager configuration.

### Usage

```bash
python test_alertmanager.py [--type {default,api,system,gpu,model}] [--severity {warning,critical}] [--service SERVICE] [--url URL]
```

### Options

- `--type`: Type of alert to test
  - Choices: default, api, system, gpu, model
  - Default: default
- `--severity`: Severity of the alert
  - Choices: warning, critical
  - Default: warning
- `--service`: Name of the service generating the alert
  - Default: vnidcard-api
- `--url`: URL of the AlertManager instance
  - Default: http://localhost:9093

### Examples

```bash
# Test a default warning alert
python test_alertmanager.py

# Test a critical API alert
python test_alertmanager.py --type api --severity critical

# Test a system alert with custom AlertManager URL
python test_alertmanager.py --type system --url http://192.168.1.100:9093
```

## Notes

- Make sure AlertManager is running when using `test_alertmanager.py`
- The Slack webhook URL is hardcoded in `test_slack_webhook.py` - update it if needed
- Each alert type will use formatting and channels based on your AlertManager configuration
