#!/bin/bash

# Vietnamese ID Card OCR - Monitoring Stack Stop Script
# This script stops the complete monitoring infrastructure

set -e

echo "🛑 Stopping Vietnamese ID Card OCR Monitoring Stack..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Stop the monitoring stack
stop_monitoring() {
    print_status "Stopping monitoring stack..."
    
    cd monitoring
    
    # Stop and remove containers
    docker-compose -f docker-compose.monitoring.yml down
    
    print_success "Monitoring stack stopped successfully!"
}

# Clean up (optional)
cleanup() {
    if [ "$1" = "--clean" ]; then
        print_status "Cleaning up monitoring data..."
        
        # Remove data directories (be careful with this!)
        read -p "Are you sure you want to delete all monitoring data? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo rm -rf monitoring/prometheus/data/*
            sudo rm -rf monitoring/grafana/data/*
            sudo rm -rf monitoring/loki/data/*
            sudo rm -rf monitoring/alertmanager/data/*
            print_success "Monitoring data cleaned up"
        else
            print_status "Keeping monitoring data"
        fi
    fi
}

# Remove Docker images (optional)
remove_images() {
    if [ "$1" = "--remove-images" ]; then
        print_status "Removing monitoring Docker images..."
        
        images=(
            "prom/prometheus"
            "grafana/grafana"
            "prom/alertmanager"
            "grafana/loki"
            "fluent/fluent-bit"
            "prom/node-exporter"
            "gcr.io/cadvisor/cadvisor"
            "nvidia/dcgm-exporter"
        )
        
        for image in "${images[@]}"; do
            if docker image inspect "$image" > /dev/null 2>&1; then
                docker rmi "$image" 2>/dev/null || true
                print_status "Removed image: $image"
            fi
        done
        
        print_success "Docker images removed"
    fi
}

# Display status
show_status() {
    echo ""
    echo "✅ Monitoring Stack Stopped"
    echo "=========================="
    echo ""
    echo "The monitoring services have been stopped."
    echo ""
    echo "💡 Available options:"
    echo "   • To restart: ./start-monitoring.sh"
    echo "   • To clean data: ./stop-monitoring.sh --clean"
    echo "   • To remove images: ./stop-monitoring.sh --remove-images"
    echo ""
}

# Main execution
main() {
    echo "Vietnamese ID Card OCR - Stop Monitoring"
    echo "======================================="
    echo ""
    
    stop_monitoring
    cleanup "$1"
    remove_images "$1"
    show_status
}

# Run main function
main "$@"
