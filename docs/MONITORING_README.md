# Vietnamese ID Card OCR - Monitoring System Documentation

## Overview

This document provides comprehensive documentation for the monitoring and observability system designed for the Vietnamese ID Card OCR API. The system provides complete visibility into application performance, system resources, and operational health.

## Architecture

### Monitoring Stack Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Prometheus    │    │     Grafana     │
│   (Port 8080)   │───▶│   (Port 9090)   │───▶│   (Port 3000)   │
│                 │    │                 │    │                 │
│ • API Metrics   │    │ • Metrics Store │    │ • Visualization │
│ • Health Checks │    │ • Alert Rules   │    │ • Dashboards    │
│ • Custom Logs   │    │ • Scraping      │    │ • Annotations   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Alertmanager   │              │
         └──────────────│   (Port 9093)   │──────────────┘
                        │                 │
                        │ • Alert Routing │
                        │ • Notifications │
                        │ • Grouping      │
                        └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Loki       │    │   Fluent Bit    │    │  Node Exporter  │
│   (Port 3100)   │◀───│                 │    │   (Port 9100)   │
│                 │    │ • Log Collection│    │                 │
│ • Log Storage   │    │ • Log Parsing   │    │ • System Metrics│
│ • Log Querying  │    │ • Log Routing   │    │ • Hardware Info │
│ • Retention     │    │ • Filtering     │    │ • OS Metrics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Metrics Collection**: FastAPI app exposes Prometheus metrics
2. **Metrics Scraping**: Prometheus scrapes metrics from all sources
3. **Log Collection**: Fluent Bit collects and processes logs
4. **Log Storage**: Loki stores and indexes logs
5. **Visualization**: Grafana queries both Prometheus and Loki
6. **Alerting**: Prometheus evaluates rules and sends alerts to Alertmanager
7. **Notifications**: Alertmanager routes alerts to various channels

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM
- 10GB free disk space
- Network access for image downloads

### Starting the Monitoring Stack

#### Linux/macOS
```bash
# Make scripts executable
chmod +x monitoring/start-monitoring.sh
chmod +x monitoring/stop-monitoring.sh
chmod +x monitoring/cleanup-logs.sh

# Start the monitoring stack
./monitoring/start-monitoring.sh
```

#### Windows
```cmd
# Start the monitoring stack
monitoring\start-monitoring.bat
```

### Accessing Services

After starting, access the services at:

- **Grafana Dashboard**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **API Documentation**: http://localhost:8080/docs

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Alert Notification Channels
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/slack/webhook
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Monitoring Configuration
PROMETHEUS_RETENTION=15d
LOKI_RETENTION=7d
LOG_LEVEL=INFO
```

### Prometheus Configuration

Located in `monitoring/prometheus/prometheus.yml`:

```yaml
# Key scrape configurations
scrape_configs:
  - job_name: 'vietnamese-id-ocr-api'
    static_configs:
      - targets: ['host.docker.internal:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s
```

### Alert Rules

Located in `monitoring/prometheus/alert-rules.yml`:

Key alert categories:
- **API Performance**: High error rates, low confidence scores
- **System Resources**: CPU, memory, disk usage
- **GPU Monitoring**: Utilization, temperature, memory
- **Model Performance**: Inference time, loading status

## Dashboards

### 1. Vietnamese ID Card API Monitoring

**Panels included:**
- Request Rate and Response Time
- Error Rate and Success Rate
- API Endpoints Performance
- Model Inference Metrics
- Confidence Score Distribution

**Key Metrics:**
- `request_count_total`: Total API requests
- `processing_time_seconds`: Request processing time
- `inference_time_seconds`: Model inference time
- `confidence_score`: Model prediction confidence

### 2. System Resource Monitoring

**Panels included:**
- CPU Usage and Load Average
- Memory Usage and Swap
- Disk Usage and I/O
- Network Traffic
- GPU Utilization and Temperature

**Key Metrics:**
- `node_cpu_seconds_total`: CPU usage by mode
- `node_memory_MemAvailable_bytes`: Available memory
- `nvidia_gpu_utilization_percent`: GPU utilization

### 3. Application Logs Dashboard

**Features:**
- Real-time log streaming
- Log level filtering
- Error pattern detection
- Custom log parsing for Vietnamese ID Card processing

## Alerting

### Alert Severity Levels

- **Critical**: System down, high error rates (>50%), GPU failures
- **Warning**: Performance degradation, resource usage >80%
- **Info**: Successful deployments, routine operations

### Notification Channels

#### Slack Integration
```bash
# Set Slack webhook URL
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

#### Telegram Integration
```bash
# Set Telegram credentials
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

#### Email Integration
Configure SMTP settings in `src/webhooks/alert_handlers.py`

### Testing Alerts

Test your notification channels:

```bash
# Test info alert
curl -X POST "http://localhost:8080/webhooks/test-alert?alert_type=test&severity=info"

# Test warning alert
curl -X POST "http://localhost:8080/webhooks/test-alert?alert_type=performance&severity=warning"

# Test critical alert
curl -X POST "http://localhost:8080/webhooks/test-alert?alert_type=system&severity=critical"
```

## Logs Management

### Log Files

The system generates several log files:

```
logs/
├── api.log          # General API logs
├── error.log        # Error-only logs
├── model.log        # Model-specific logs
├── metrics.log      # Metrics-related logs
└── cleanup_report_* # Cleanup operation reports
```

### Log Rotation

Automatic log rotation based on:
- **Size**: Files >100MB are rotated
- **Time**: Daily rotation at 2 AM (if scheduled)
- **Retention**: 30 days for logs, 15 days for metrics

### Manual Log Management

```bash
# Rotate logs manually
./monitoring/cleanup-logs.sh

# Create backup before cleanup
./monitoring/cleanup-logs.sh --backup

# Windows
monitoring\cleanup-logs.bat --backup
```

## Metrics Reference

### API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `request_count_total` | Counter | Total number of API requests |
| `processing_time_seconds` | Histogram | Time spent processing requests |
| `error_count_total` | Counter | Total number of errors |
| `success_count_total` | Counter | Total number of successful requests |
| `inference_time_seconds` | Histogram | Model inference time |
| `confidence_score` | Histogram | Model prediction confidence |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cpu_usage_percent` | Gauge | Current CPU usage percentage |
| `memory_usage_percent` | Gauge | Current memory usage percentage |
| `disk_usage_percent` | Gauge | Current disk usage percentage |
| `gpu_usage_percent` | Gauge | GPU utilization percentage |
| `gpu_temperature_celsius` | Gauge | GPU temperature in Celsius |

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

**Symptoms**: Cannot access Grafana/Prometheus
**Solutions**:
```bash
# Check service status
docker-compose -f monitoring/docker-compose.monitoring.yml ps

# Check service logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs grafana

# Restart specific service
docker-compose -f monitoring/docker-compose.monitoring.yml restart grafana
```

#### 2. No Metrics in Grafana

**Symptoms**: Empty dashboards
**Solutions**:
- Verify Prometheus is scraping targets: http://localhost:9090/targets
- Check API is exposing metrics: http://localhost:8080/metrics
- Verify Grafana data source configuration

#### 3. Alerts Not Firing

**Symptoms**: No alert notifications
**Solutions**:
- Check Prometheus alert rules: http://localhost:9090/alerts
- Verify Alertmanager configuration: http://localhost:9093
- Test webhook endpoints: http://localhost:8080/webhooks/health

#### 4. High Disk Usage

**Symptoms**: Monitoring services consuming too much disk
**Solutions**:
```bash
# Check data directory sizes
du -sh monitoring/*/data

# Clean old data
./monitoring/cleanup-logs.sh --backup

# Reduce retention periods in configuration
```

### Log Analysis

#### Finding Errors
```bash
# Search for errors in logs
grep "ERROR" logs/error.log

# Count errors by hour
grep "$(date +%Y-%m-%d)" logs/error.log | cut -d' ' -f2 | cut -d':' -f1 | sort | uniq -c
```

#### Performance Analysis
```bash
# Find slow requests
grep "processing_time" logs/metrics.log | awk '$NF > 5'

# Average confidence scores
grep "confidence" logs/model.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

## Performance Tuning

### Resource Allocation

Adjust Docker Compose resource limits:

```yaml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Data Retention

Balance storage vs. historical data:

```yaml
# Prometheus retention
command:
  - '--storage.tsdb.retention.time=15d'
  - '--storage.tsdb.retention.size=5GB'

# Loki retention
limits_config:
  retention_period: 168h  # 7 days
```

### Scrape Intervals

Optimize based on needs:

```yaml
scrape_configs:
  - job_name: 'api'
    scrape_interval: 15s    # High frequency for API
  - job_name: 'system'
    scrape_interval: 30s    # Lower frequency for system
```

## Security Considerations

### Access Control

1. **Network Security**: Use Docker networks for service isolation
2. **Authentication**: Enable Grafana authentication for production
3. **API Keys**: Secure API keys using environment variables
4. **Firewall**: Restrict external access to monitoring ports

### Data Privacy

1. **Log Sanitization**: Remove sensitive data from logs
2. **Metric Labels**: Avoid including PII in metric labels
3. **Retention**: Implement appropriate data retention policies

## Maintenance

### Regular Tasks

#### Daily
- Monitor alert notifications
- Check system resource usage
- Review error logs

#### Weekly
- Review dashboard performance
- Update alert thresholds if needed
- Check log rotation

#### Monthly
- Update monitoring stack images
- Review and update alert rules
- Analyze performance trends
- Backup monitoring configuration

### Updates

```bash
# Update monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml pull
docker-compose -f docker-compose.monitoring.yml up -d
```

## Integration with CI/CD

### GitHub Actions Integration

```yaml
# .github/workflows/monitoring.yml
name: Deploy Monitoring
on:
  push:
    paths:
      - 'monitoring/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy monitoring stack
        run: |
          chmod +x monitoring/start-monitoring.sh
          ./monitoring/start-monitoring.sh
```

### Health Checks

Monitor deployment success:

```bash
# API health check
curl -f http://localhost:8080/health

# Monitoring stack health
curl -f http://localhost:9090/-/healthy
curl -f http://localhost:3000/api/health
```

## Support and Contributing

### Getting Help

1. Check the troubleshooting section
2. Review logs for error messages
3. Consult Prometheus/Grafana documentation
4. Create an issue with relevant logs and configuration

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new monitoring features
4. Update documentation
5. Submit a pull request

## Appendix

### Useful Commands

```bash
# View all metrics
curl http://localhost:8080/metrics

# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=up'

# View Grafana API
curl -u admin:admin http://localhost:3000/api/dashboards/home

# Alertmanager API
curl http://localhost:9093/api/v1/alerts
```

### Configuration Templates

#### Custom Alert Rule
```yaml
groups:
  - name: custom.rules
    rules:
      - alert: CustomMetricHigh
        expr: custom_metric > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Custom metric is high"
          description: "Custom metric has been above 100 for 5 minutes"
```

#### Custom Dashboard Panel
```json
{
  "title": "Custom Metric",
  "type": "timeseries",
  "targets": [
    {
      "expr": "custom_metric",
      "legendFormat": "Custom Metric"
    }
  ]
}
```

This documentation provides a complete guide to deploying, configuring, and maintaining the Vietnamese ID Card OCR monitoring system. For additional help, refer to the official documentation of each component or contact the development team.
