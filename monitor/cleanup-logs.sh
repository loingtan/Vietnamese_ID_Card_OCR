#!/bin/bash

# Vietnamese ID Card OCR - Log Cleanup and Rotation Script
# This script manages log rotation and cleanup for the monitoring system

set -e

# Configuration
LOG_DIR="logs"
PROMETHEUS_DATA_DIR="monitoring/prometheus/data"
GRAFANA_DATA_DIR="monitoring/grafana/data"
LOKI_DATA_DIR="monitoring/loki/data"
ALERTMANAGER_DATA_DIR="monitoring/alertmanager/data"

# Retention settings (days)
LOG_RETENTION_DAYS=30
METRICS_RETENTION_DAYS=15
BACKUP_RETENTION_DAYS=7

# Size limits (MB)
MAX_LOG_SIZE=100
MAX_METRICS_SIZE=1000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Rotate application logs
rotate_application_logs() {
    print_status "Rotating application logs..."
    
    if [ ! -d "$LOG_DIR" ]; then
        print_warning "Log directory $LOG_DIR does not exist"
        return
    fi
    
    cd "$LOG_DIR"
    
    # Rotate each log file
    for log_file in api.log error.log model.log metrics.log; do
        if [ -f "$log_file" ]; then
            # Check file size
            size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo 0)
            size_mb=$((size / 1024 / 1024))
            
            if [ $size_mb -gt $MAX_LOG_SIZE ]; then
                timestamp=$(date +"%Y%m%d_%H%M%S")
                mv "$log_file" "${log_file}.${timestamp}"
                touch "$log_file"
                gzip "${log_file}.${timestamp}"
                print_status "Rotated $log_file (${size_mb}MB)"
            fi
        fi
    done
    
    cd ..
}

# Clean old log files
clean_old_logs() {
    print_status "Cleaning old log files..."
    
    if [ ! -d "$LOG_DIR" ]; then
        return
    fi
    
    # Remove old rotated logs
    find "$LOG_DIR" -name "*.log.*" -type f -mtime +$LOG_RETENTION_DAYS -delete
    
    # Remove old compressed logs
    find "$LOG_DIR" -name "*.gz" -type f -mtime +$LOG_RETENTION_DAYS -delete
    
    deleted_count=$(find "$LOG_DIR" -name "*.log.*" -o -name "*.gz" | wc -l)
    print_success "Cleaned old log files (${deleted_count} files removed)"
}

# Clean Prometheus data
clean_prometheus_data() {
    print_status "Cleaning old Prometheus data..."
    
    if [ ! -d "$PROMETHEUS_DATA_DIR" ]; then
        print_warning "Prometheus data directory does not exist"
        return
    fi
    
    # Remove old WAL and block data (Prometheus handles this mostly)
    # Only clean if data is older than retention period
    find "$PROMETHEUS_DATA_DIR" -name "*.tmp" -type f -mtime +1 -delete 2>/dev/null || true
    
    # Check data directory size
    if command -v du &> /dev/null; then
        size=$(du -sm "$PROMETHEUS_DATA_DIR" 2>/dev/null | cut -f1)
        if [ $size -gt $MAX_METRICS_SIZE ]; then
            print_warning "Prometheus data size (${size}MB) exceeds limit (${MAX_METRICS_SIZE}MB)"
            print_warning "Consider reducing Prometheus retention period"
        fi
    fi
    
    print_success "Prometheus data cleanup completed"
}

# Backup critical data
backup_data() {
    print_status "Creating backup of critical data..."
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Grafana dashboards and configuration
    if [ -d "$GRAFANA_DATA_DIR" ]; then
        cp -r monitoring/grafana/provisioning "$backup_dir/grafana_provisioning" 2>/dev/null || true
        cp -r monitoring/grafana/dashboards "$backup_dir/grafana_dashboards" 2>/dev/null || true
    fi
    
    # Backup Prometheus configuration
    if [ -f "monitoring/prometheus/prometheus.yml" ]; then
        cp monitoring/prometheus/prometheus.yml "$backup_dir/" 2>/dev/null || true
        cp monitoring/prometheus/alert-rules.yml "$backup_dir/" 2>/dev/null || true
    fi
    
    # Backup Alertmanager configuration
    if [ -f "monitoring/alertmanager/alertmanager.yml" ]; then
        cp monitoring/alertmanager/alertmanager.yml "$backup_dir/" 2>/dev/null || true
    fi
    
    # Backup Loki configuration
    if [ -f "monitoring/loki/loki-config.yml" ]; then
        cp monitoring/loki/loki-config.yml "$backup_dir/" 2>/dev/null || true
    fi
    
    # Compress backup
    tar -czf "backups/monitoring_backup_$(date +%Y%m%d_%H%M%S).tar.gz" -C "$backup_dir" . 2>/dev/null || true
    rm -rf "$backup_dir"
    
    print_success "Backup created in backups/ directory"
}

# Clean old backups
clean_old_backups() {
    print_status "Cleaning old backups..."
    
    if [ -d "backups" ]; then
        find backups -name "*.tar.gz" -type f -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        deleted_count=$(find backups -name "*.tar.gz" 2>/dev/null | wc -l)
        print_success "Old backups cleaned (keeping ${deleted_count} recent backups)"
    fi
}

# Generate cleanup report
generate_report() {
    print_status "Generating cleanup report..."
    
    report_file="logs/cleanup_report_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Vietnamese ID Card OCR - Cleanup Report"
        echo "======================================="
        echo "Date: $(date)"
        echo ""
        
        echo "Log Directory Status:"
        if [ -d "$LOG_DIR" ]; then
            du -sh "$LOG_DIR" 2>/dev/null || echo "Size calculation failed"
            ls -la "$LOG_DIR" 2>/dev/null || echo "Directory listing failed"
        else
            echo "Log directory does not exist"
        fi
        echo ""
        
        echo "Monitoring Data Status:"
        for dir in "$PROMETHEUS_DATA_DIR" "$GRAFANA_DATA_DIR" "$LOKI_DATA_DIR" "$ALERTMANAGER_DATA_DIR"; do
            if [ -d "$dir" ]; then
                echo "$(basename $dir): $(du -sh $dir 2>/dev/null | cut -f1)"
            else
                echo "$(basename $dir): Directory does not exist"
            fi
        done
        echo ""
        
        echo "Disk Usage:"
        df -h . 2>/dev/null || echo "Disk usage calculation failed"
        echo ""
        
        echo "Cleanup Settings:"
        echo "  Log retention: $LOG_RETENTION_DAYS days"
        echo "  Metrics retention: $METRICS_RETENTION_DAYS days"
        echo "  Backup retention: $BACKUP_RETENTION_DAYS days"
        echo "  Max log size: $MAX_LOG_SIZE MB"
        echo "  Max metrics size: $MAX_METRICS_SIZE MB"
        
    } > "$report_file"
    
    print_success "Cleanup report saved to $report_file"
}

# Main cleanup function
main() {
    echo "Vietnamese ID Card OCR - Log Cleanup and Rotation"
    echo "================================================="
    echo ""
    
    # Create necessary directories
    mkdir -p logs backups
    
    # Perform cleanup tasks
    rotate_application_logs
    clean_old_logs
    clean_prometheus_data
    
    # Backup if requested
    if [ "$1" = "--backup" ]; then
        backup_data
    fi
    
    clean_old_backups
    generate_report
    
    echo ""
    print_success "Cleanup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  • Review cleanup report in logs/ directory"
    echo "  • Schedule this script with cron for automatic cleanup"
    echo "  • Example cron entry (daily at 2 AM):"
    echo "    0 2 * * * /path/to/cleanup-logs.sh --backup"
    echo ""
}

# Show usage if help requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Vietnamese ID Card OCR - Log Cleanup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --backup    Create backup before cleanup"
    echo "  --help, -h  Show this help message"
    echo ""
    echo "This script performs the following tasks:"
    echo "  • Rotates large log files"
    echo "  • Removes old log files based on retention policy"
    echo "  • Cleans old Prometheus data"
    echo "  • Creates backups of monitoring configuration"
    echo "  • Removes old backups"
    echo "  • Generates cleanup report"
    echo ""
    exit 0
fi

# Run main function
main "$@"
