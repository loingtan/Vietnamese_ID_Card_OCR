#!/bin/bash

# Vietnamese ID Card OCR - Complete Stack Startup Script
# This script starts the complete application and monitoring infrastructure

set -e

echo "🚀 Starting Vietnamese ID Card OCR Complete Stack..."

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

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Navigate to deployment directory
navigate_to_deployment() {
    print_status "Navigating to deployment directory..."
    cd "$(dirname "$0")/../deployment/docker"
    print_success "Changed to deployment directory"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p ../../logs
    print_success "Directories created successfully"
}

# Start the complete stack
start_stack() {
    print_status "Starting complete stack with monitoring..."
    
    # Pull latest images
    print_status "Pulling latest Docker images..."
    docker-compose pull
    
    # Start services with monitoring profile
    print_status "Starting services with monitoring profile..."
    docker-compose --profile monitoring up -d
    
    print_success "Complete stack started successfully!"
}

# Wait for services to start
wait_for_services() {
    print_status "Waiting for services to start..."
    sleep 15
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    # Check API
    if curl -f -s "http://localhost:8080/health" > /dev/null 2>&1; then
        print_success "API is running on port 8080"
    else
        print_warning "API might not be ready yet (port 8080)"
    fi
    
    # Check Prometheus
    if curl -f -s "http://localhost:9090" > /dev/null 2>&1; then
        print_success "Prometheus is running on port 9090"
    else
        print_warning "Prometheus might not be ready yet (port 9090)"
    fi
    
    # Check Grafana
    if curl -f -s "http://localhost:3000" > /dev/null 2>&1; then
        print_success "Grafana is running on port 3000"
    else
        print_warning "Grafana might not be ready yet (port 3000)"
    fi
    
    # Check Alertmanager
    if curl -f -s "http://localhost:9093" > /dev/null 2>&1; then
        print_success "Alertmanager is running on port 9093"
    else
        print_warning "Alertmanager might not be ready yet (port 9093)"
    fi
    
    # Check Loki
    if curl -f -s "http://localhost:3100/ready" > /dev/null 2>&1; then
        print_success "Loki is running on port 3100"
    else
        print_warning "Loki might not be ready yet (port 3100)"
    fi
}

# Display access information
display_access_info() {
    echo
    echo "🎉 Complete Stack Successfully Started!"
    echo "======================================"
    echo
    echo "🚀 Access your services:"
    echo "   • API Application:      http://localhost:8080"
    echo "   • Streamlit UI:         http://localhost:8501"
    echo "   • API Metrics:          http://localhost:8000"
    echo
    echo "📊 Monitoring Services:"
    echo "   • Grafana Dashboard:    http://localhost:3000"
    echo "     - Username: admin"
    echo "     - Password: vnidcard123"
    echo
    echo "   • Prometheus:           http://localhost:9090"
    echo "   • Alertmanager:         http://localhost:9093"
    echo "   • Loki:                 http://localhost:3100"
    echo
    echo "📈 Pre-configured Dashboards:"
    echo "   • Vietnamese ID Card API Monitoring"
    echo "   • System Resource Monitoring"
    echo "   • Application Logs Dashboard"
    echo
    echo "🔔 Alerts are configured for:"
    echo "   • High API error rates"
    echo "   • Low confidence scores"
    echo "   • High system resource usage"
    echo "   • GPU performance issues"
    echo
    echo "💡 To stop the complete stack:"
    echo "   ./stop-monitoring.sh"
    echo
}

# Main execution
main() {
    check_dependencies
    navigate_to_deployment
    create_directories
    start_stack
    wait_for_services
    check_service_health
    display_access_info
}

# Run main function
main "$@"
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p monitoring/prometheus/data
    mkdir -p monitoring/grafana/data
    mkdir -p monitoring/loki/data
    mkdir -p monitoring/alertmanager/data
    
    # Set proper permissions for Grafana
    chmod 777 monitoring/grafana/data
    
    print_success "Directories created successfully"
}

# Start the monitoring stack
start_monitoring() {
    print_status "Starting monitoring stack..."
    
    cd monitoring
    
    # Pull latest images
    print_status "Pulling latest Docker images..."
    docker-compose -f docker-compose.monitoring.yml pull
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.monitoring.yml up -d
    
    print_success "Monitoring stack started successfully!"
}

# Check service health
check_services() {
    print_status "Checking service health..."
    
    # Wait a bit for services to start
    sleep 10
    
    # Check each service
    services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "loki:3100")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -f -s "http://localhost:$port" > /dev/null 2>&1; then
            print_success "$name is running on port $port"
        else
            print_warning "$name might not be ready yet (port $port)"
        fi
    done
}

# Display access information
show_access_info() {
    echo ""
    echo "🎉 Monitoring Stack Successfully Started!"
    echo "=========================================="
    echo ""
    echo "📊 Access your monitoring services:"
    echo "   • Grafana Dashboard:    http://localhost:3000"
    echo "     - Username: admin"
    echo "     - Password: admin"
    echo ""
    echo "   • Prometheus:           http://localhost:9090"
    echo "   • Alertmanager:         http://localhost:9093"
    echo "   • Loki:                 http://localhost:3100"
    echo ""
    echo "📈 Pre-configured Dashboards:"
    echo "   • Vietnamese ID Card API Monitoring"
    echo "   • System Resource Monitoring"
    echo "   • Application Logs Dashboard"
    echo ""
    echo "🔔 Alerts are configured for:"
    echo "   • High API error rates"
    echo "   • Low confidence scores"
    echo "   • High system resource usage"
    echo "   • GPU performance issues"
    echo ""
    echo "💡 To stop the monitoring stack:"
    echo "   ./stop-monitoring.sh"
    echo ""
}

# Main execution
main() {
    echo "Vietnamese ID Card OCR - Monitoring Stack"
    echo "========================================="
    echo ""
    
    check_dependencies
    create_directories
    start_monitoring
    check_services
    show_access_info
}

# Run main function
main "$@"
