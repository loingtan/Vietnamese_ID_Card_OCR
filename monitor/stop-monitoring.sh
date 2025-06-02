#!/bin/bash

# Vietnamese ID Card OCR - Complete Stack Stop Script
# This script stops the complete application and monitoring infrastructure

set -e

echo "🛑 Stopping Vietnamese ID Card OCR Complete Stack..."

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

# Navigate to deployment directory
navigate_to_deployment() {
    print_status "Navigating to deployment directory..."
    cd "$(dirname "$0")/../deployment/docker"
    print_success "Changed to deployment directory"
}

# Stop the complete stack
stop_stack() {
    print_status "Stopping complete stack..."
    
    # Stop and remove containers
    docker-compose down
    
    print_success "Complete stack stopped successfully!"
}

# Clean up (optional)
cleanup() {
    if [ "$1" = "--clean" ]; then
        print_status "Cleaning up data..."
        
        read -p "Are you sure you want to delete all data volumes? (y/N): " clean_confirm
        if [[ $clean_confirm =~ ^[Yy]$ ]]; then
            docker-compose down -v
            docker system prune -f
            print_success "All data cleaned up"
        else
            print_status "Keeping data volumes"
        fi
    fi
}

# Display completion information
display_completion_info() {
    echo
    echo "✅ Complete Stack Stopped Successfully!"
    echo
    echo "💡 To restart the stack:"
    echo "   ./start-monitoring.sh"
    echo
    echo "🧹 To stop and clean all data:"
    echo "   ./stop-monitoring.sh --clean"
    echo
}

# Main execution
main() {
    navigate_to_deployment
    stop_stack
    cleanup "$1"
    display_completion_info
}

# Run main function
main "$@"
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
