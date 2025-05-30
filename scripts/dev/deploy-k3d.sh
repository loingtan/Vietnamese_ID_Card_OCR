#!/bin/bash

# K3D Deployment Script for Vietnamese ID Card OCR Application
# This script sets up a complete k3d cluster with registry and deploys the application

set -e

echo "🚀 Starting k3d deployment for Vietnamese ID Card OCR Application"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="vnidcard-cluster"
REGISTRY_NAME="vnidcard-registry"
REGISTRY_PORT="5000"
APP_NAME="vnidcard-app"
NAMESPACE="default"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if k3d is installed
check_k3d() {
    if ! command -v k3d &> /dev/null; then
        print_error "k3d is not installed. Please install k3d first."
        echo "Visit: https://k3d.io/v5.6.0/#installation"
        exit 1
    fi
    print_status "k3d is installed: $(k3d version)"
}

# Check if Docker is running
check_docker() {
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_status "Docker is running"
}

# Clean up existing cluster and registry
cleanup() {
    print_status "Cleaning up existing cluster and registry..."
    k3d cluster delete $CLUSTER_NAME 2>/dev/null || true
    k3d registry delete $REGISTRY_NAME 2>/dev/null || true
    docker network rm vnidcard-net 2>/dev/null || true
}

# Create k3d cluster with registry
create_cluster() {
    print_status "Creating k3d cluster with configuration..."
    
    # Create the cluster using config file
    k3d cluster create --config k3d-config.yaml
    
    print_status "Cluster created successfully!"
    
    # Wait for cluster to be ready
    print_status "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=300s
    
    print_status "Cluster is ready!"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t $APP_NAME:latest .
    
    # Tag for registry
    docker tag $APP_NAME:latest localhost:$REGISTRY_PORT/$APP_NAME:latest
    
    print_status "Pushing image to local registry..."
    docker push localhost:$REGISTRY_PORT/$APP_NAME:latest
    
    print_status "Image pushed successfully!"
}

# Deploy application
deploy_app() {
    print_status "Deploying application to Kubernetes..."
    
    # Apply deployment
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$APP_NAME
    
    print_status "Application deployed successfully!"
}

# Get application info
get_app_info() {
    print_status "Getting application information..."
    
    echo ""
    echo "=== CLUSTER INFORMATION ==="
    kubectl cluster-info
    
    echo ""
    echo "=== NODES ==="
    kubectl get nodes -o wide
    
    echo ""
    echo "=== PODS ==="
    kubectl get pods -o wide
    
    echo ""
    echo "=== SERVICES ==="
    kubectl get services -o wide
    
    echo ""
    echo "=== DEPLOYMENTS ==="
    kubectl get deployments -o wide
    
    echo ""
    print_status "Application URLs:"
    echo "- Streamlit App: http://localhost:8501"
    echo "- API (if available): http://localhost:8080"
    echo "- Registry: http://localhost:5000"
    
    echo ""
    print_status "Useful commands:"
    echo "- View logs: kubectl logs -l app=$APP_NAME -f"
    echo "- Scale app: kubectl scale deployment $APP_NAME --replicas=3"
    echo "- Delete cluster: k3d cluster delete $CLUSTER_NAME"
    echo "- Access cluster: export KUBECONFIG=\$(k3d kubeconfig write $CLUSTER_NAME)"
}

# Main execution
main() {
    print_status "Starting deployment process..."
    
    check_k3d
    check_docker
    
    # Ask for confirmation
    read -p "This will delete existing cluster '$CLUSTER_NAME'. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled."
        exit 0
    fi
    
    cleanup
    create_cluster
    build_and_push_image
    deploy_app
    get_app_info
    
    print_status "Deployment completed successfully! 🎉"
}

# Run main function
main "$@"
