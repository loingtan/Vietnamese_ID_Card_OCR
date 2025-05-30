# Vietnamese ID Card OCR - Monitoring System Deployment Guide

This guide provides step-by-step instructions for deploying the complete monitoring stack for the Vietnamese ID Card OCR system.

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 4GB, Recommended 8GB
- **Disk**: 10GB free space
- **Docker**: Docker Desktop with Docker Compose
- **Network**: Internet access for downloading images

### Software Dependencies
- Docker Desktop (latest version)
- PowerShell (Windows) or Bash (Linux/macOS)
- Git (for cloning and updates)

## Installation Steps

### Step 1: Verify Prerequisites

Run the prerequisite check script:

**Windows (PowerShell):**
```powershell
.\check-prerequisites.ps1
```

**Linux/macOS:**
```bash
chmod +x check-prerequisites.sh
./check-prerequisites.sh
```

### Step 2: Configure Environment

Create a `.env` file in the project root:

```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Monitoring Configuration
PROMETHEUS_RETENTION=15d
LOKI_RETENTION=7d
LOG_LEVEL=INFO

# Alert Notification Channels (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/slack/webhook
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Step 3: Start the Monitoring Stack

**Windows:**
```powershell
cd monitoring
.\start-monitoring.bat
```

**Linux/macOS:**
```bash
cd monitoring
chmod +x start-monitoring.sh
./start-monitoring.sh
```

### Step 4: Verify Deployment

After startup, verify all services are running:

1. **Grafana**: http://localhost:3000 (admin/admin)
2. **Prometheus**: http://localhost:9090
3. **Alertmanager**: http://localhost:9093
4. **API Health**: http://localhost:8080/health

### Step 5: Import Dashboards

Dashboards are automatically provisioned, but you can also import manually:

1. Go to Grafana → Dashboards → Import
2. Upload the JSON files from `monitoring/grafana/dashboards/`
3. Configure data sources if needed

## Service Details

### Prometheus (Port 9090)
- **Purpose**: Metrics collection and storage
- **Retention**: 15 days (configurable)
- **Scrape Interval**: 15 seconds
- **Storage**: Local Docker volume

### Grafana (Port 3000)
- **Purpose**: Visualization and dashboards
- **Default Login**: admin/admin
- **Dashboards**: 3 pre-configured dashboards
- **Data Sources**: Prometheus and Loki

### Alertmanager (Port 9093)
- **Purpose**: Alert routing and notifications
- **Channels**: Slack, Telegram, Email, Webhook
- **Grouping**: By severity and service
- **Retry Logic**: Exponential backoff

### Loki (Port 3100)
- **Purpose**: Log aggregation and storage
- **Retention**: 7 days (configurable)
- **Compression**: Gzip compression enabled
- **Indexing**: Optimized for time-based queries

### Fluent Bit
- **Purpose**: Log collection and processing
- **Sources**: Application logs, container logs, system logs
- **Parsing**: Custom parsers for Vietnamese ID Card logs
- **Filtering**: Confidence score filtering with Lua

### Node Exporter (Port 9100)
- **Purpose**: System metrics collection
- **Metrics**: CPU, memory, disk, network
- **Frequency**: 15-second intervals
- **Scope**: Host system metrics

### cAdvisor (Port 8080)
- **Purpose**: Container metrics collection
- **Metrics**: Container resource usage
- **Scope**: All Docker containers
- **Integration**: Prometheus scraping

## Configuration Files

### Key Configuration Files
- `monitoring/docker-compose.monitoring.yml` - Main orchestration
- `monitoring/prometheus/prometheus.yml` - Prometheus configuration
- `monitoring/prometheus/alert-rules.yml` - Alert definitions
- `monitoring/alertmanager/alertmanager.yml` - Alert routing
- `monitoring/loki/loki-config.yml` - Log storage configuration
- `monitoring/fluent-bit/fluent-bit.conf` - Log collection
- `monitoring/grafana/provisioning/` - Dashboard provisioning

### Customization

#### Modify Retention Periods
```yaml
# prometheus.yml
global:
  external_labels:
    monitor: 'vnid-card-monitor'
  
# Change retention
command:
  - '--storage.tsdb.retention.time=30d'  # Increase to 30 days
```

#### Add Custom Alerts
```yaml
# alert-rules.yml
groups:
  - name: custom.rules
    rules:
      - alert: CustomMetricAlert
        expr: your_custom_metric > threshold
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Custom alert triggered"
```

#### Configure Slack Notifications
```yaml
# alertmanager.yml
receivers:
  - name: 'slack-notifications'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'Vietnamese ID Card OCR Alert'
```

## Monitoring Dashboards

### 1. API Monitoring Dashboard
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: Error percentage over time
- **Inference Time**: Model processing time
- **Confidence Scores**: Distribution and trends

### 2. System Monitoring Dashboard
- **CPU Usage**: Per-core and average
- **Memory Usage**: Available vs used
- **Disk Usage**: Usage by filesystem
- **Network I/O**: Bytes sent/received
- **GPU Metrics**: Utilization, memory, temperature

### 3. Logs Dashboard
- **Log Stream**: Real-time log viewing
- **Error Logs**: Filtered error messages
- **Log Levels**: Distribution by severity
- **Search**: Full-text log search
- **Parsing**: Structured log viewing

## Alert Configuration

### Alert Rules Summary

| Alert | Condition | Severity | Action |
|-------|-----------|----------|---------|
| APIHighErrorRate | Error rate > 50% | Critical | All channels |
| APILowConfidence | Confidence < 0.6 | Warning | Slack + Email |
| APIHighLatency | P95 latency > 30s | Warning | Slack |
| SystemHighCPU | CPU > 80% | Warning | Slack |
| SystemHighMemory | Memory > 90% | Critical | All channels |
| SystemHighDisk | Disk > 85% | Warning | Slack |
| GPUHighTemp | GPU temp > 85°C | Critical | All channels |
| ContainerDown | Container not running | Critical | All channels |

### Testing Alerts

Test your alert configuration:

```powershell
# Test different severity levels
Invoke-RestMethod -Uri "http://localhost:8080/webhooks/test-alert?alert_type=test&severity=info" -Method POST
Invoke-RestMethod -Uri "http://localhost:8080/webhooks/test-alert?alert_type=performance&severity=warning" -Method POST
Invoke-RestMethod -Uri "http://localhost:8080/webhooks/test-alert?alert_type=system&severity=critical" -Method POST
```

## Maintenance

### Daily Operations

#### Check System Health
```powershell
# View running containers
docker ps

# Check service logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs --tail=50

# Monitor resource usage
docker stats
```

#### Log Management
```powershell
# Rotate logs manually
.\monitoring\cleanup-logs.bat

# Create backup
.\monitoring\cleanup-logs.bat --backup
```

### Weekly Operations

#### Update Images
```powershell
cd monitoring
docker-compose -f docker-compose.monitoring.yml pull
docker-compose -f docker-compose.monitoring.yml up -d
```

#### Review Alerts
1. Check Alertmanager for fired alerts
2. Review alert frequency and accuracy
3. Adjust thresholds if needed
4. Update notification channels

### Monthly Operations

#### Performance Review
1. Analyze dashboard metrics trends
2. Review log patterns and errors
3. Optimize resource allocation
4. Plan capacity upgrades

#### Backup Configuration
```powershell
# Create configuration backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = "backups\monitoring_config_$timestamp"
New-Item -ItemType Directory -Path $backupPath -Force

# Copy configuration files
Copy-Item "monitoring\prometheus\*.yml" $backupPath
Copy-Item "monitoring\alertmanager\*.yml" $backupPath
Copy-Item "monitoring\loki\*.yml" $backupPath
Copy-Item "monitoring\grafana\provisioning" $backupPath -Recurse

# Compress backup
Compress-Archive -Path $backupPath -DestinationPath "$backupPath.zip"
Remove-Item $backupPath -Recurse
```

## Troubleshooting

### Common Issues

#### Issue 1: Grafana Shows "No Data"
**Symptoms**: Empty dashboards, no metrics
**Solutions**:
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify API metrics endpoint: http://localhost:8080/metrics
3. Check Grafana data source configuration
4. Restart Prometheus service

#### Issue 2: High Memory Usage
**Symptoms**: System slowdown, out of memory errors
**Solutions**:
1. Reduce Prometheus retention period
2. Limit container memory in docker-compose.yml
3. Clean old log files
4. Increase system RAM

#### Issue 3: Alerts Not Triggering
**Symptoms**: No alert notifications despite issues
**Solutions**:
1. Check Prometheus alert rules: http://localhost:9090/alerts
2. Verify Alertmanager configuration
3. Test webhook endpoints
4. Check notification channel configuration

#### Issue 4: Log Collection Issues
**Symptoms**: Missing logs in Grafana
**Solutions**:
1. Check Fluent Bit logs for errors
2. Verify log file permissions
3. Check Loki ingestion endpoint
4. Review Fluent Bit configuration

### Recovery Procedures

#### Service Recovery
```powershell
# Restart specific service
docker-compose -f monitoring/docker-compose.monitoring.yml restart prometheus

# Restart entire stack
docker-compose -f monitoring/docker-compose.monitoring.yml down
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

#### Data Recovery
```powershell
# Restore from backup
$backupFile = "backups\monitoring_backup_latest.zip"
Expand-Archive -Path $backupFile -DestinationPath "temp_restore"

# Copy configuration files back
Copy-Item "temp_restore\*.yml" "monitoring\prometheus\"
Copy-Item "temp_restore\alertmanager.yml" "monitoring\alertmanager\"

# Restart services
docker-compose -f monitoring/docker-compose.monitoring.yml restart
```

## Security

### Access Control
1. Change default Grafana password immediately
2. Use strong passwords for all services
3. Limit network access to monitoring ports
4. Enable HTTPS in production

### Data Protection
1. Avoid logging sensitive information
2. Sanitize metrics labels
3. Implement log retention policies
4. Secure backup storage

### Network Security
1. Use Docker networks for service isolation
2. Configure firewall rules
3. Enable authentication for external access
4. Use VPN for remote monitoring

## Performance Optimization

### Resource Allocation
```yaml
# Optimize container resources
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Storage Optimization
- Use SSD storage for better performance
- Monitor disk usage regularly
- Implement data compression
- Configure appropriate retention periods

### Network Optimization
- Use local networks for service communication
- Optimize scrape intervals
- Implement metric filtering
- Use compression for data transfer

This deployment guide provides everything needed to successfully deploy and maintain the Vietnamese ID Card OCR monitoring system. Follow the steps carefully and refer to the troubleshooting section for any issues.
