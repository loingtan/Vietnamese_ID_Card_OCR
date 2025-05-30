# K3D Deployment Script for Vietnamese ID Card OCR Application (PowerShell version)
# This script sets up a complete k3d cluster with registry and deploys the application

param(
    [switch]$Force,
    [switch]$SkipBuild,
    [string]$ClusterName = "vnidcard-cluster",
    [string]$RegistryName = "vnidcard-registry",
    [int]$RegistryPort = 5001,
    [string]$AppName = "vnidcard-app"
)

# Configuration
$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-K3d {
    Write-Status "Checking k3d installation..."
    try {
        $version = k3d version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Status "k3d is installed: $version"
            return $true
        }
    }
    catch {
        Write-Error "k3d is not installed. Please install k3d first."
        Write-Host "Visit: https://k3d.io/v5.6.0/#installation"
        return $false
    }
}

function Test-Docker {
    Write-Status "Checking Docker status..."
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Docker is running"
            return $true
        }
    }
    catch {
        Write-Error "Docker is not running. Please start Docker first."
        return $false
    }
}

function Remove-ExistingResources {
    Write-Status "Cleaning up existing cluster and registry..."
    
    try {
        k3d cluster delete $ClusterName 2>$null
    }
    catch {
        # Ignore errors if cluster doesn't exist
    }
    
    try {
        k3d registry delete $RegistryName 2>$null
    }
    catch {
        # Ignore errors if registry doesn't exist
    }
    
    try {
        docker network rm vnidcard-net 2>$null
    }
    catch {
        # Ignore errors if network doesn't exist
    }
}

function New-K3dCluster {
    Write-Status "Creating k3d cluster with configuration..."
    
    try {
        # Create the cluster using config file
        k3d cluster create --config k3d-config.yaml
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create k3d cluster"
        }
        
        Write-Status "Cluster created successfully!"
        
        # Wait for cluster to be ready
        Write-Status "Waiting for cluster to be ready..."
        kubectl wait --for=condition=Ready nodes --all --timeout=300s
        
        if ($LASTEXITCODE -ne 0) {
            throw "Cluster nodes not ready within timeout"
        }
        
        Write-Status "Cluster is ready!"
    }
    catch {
        Write-Error "Failed to create cluster: $($_.Exception.Message)"
        throw
    }
}

function Build-AndPushImage {
    if ($SkipBuild) {
        Write-Status "Skipping image build as requested"
        return
    }
    
    Write-Status "Building Docker image..."
    
    try {
        # Build the image
        docker build -t "$($AppName):latest" .
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build Docker image"
        }
        
        # Tag for registry
        docker tag "$($AppName):latest" "localhost:$($RegistryPort)/$($AppName):latest"
        
        Write-Status "Pushing image to local registry..."
        docker push "localhost:$($RegistryPort)/$($AppName):latest"
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to push image to registry"
        }
        
        Write-Status "Image pushed successfully!"
    }
    catch {
        Write-Error "Failed to build and push image: $($_.Exception.Message)"
        throw
    }
}

function Deploy-Application {
    Write-Status "Deploying application to Kubernetes..."
    
    try {
        # Apply deployment and service
        kubectl apply -f deployment.yaml
        kubectl apply -f service.yaml
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to apply Kubernetes manifests"
        }
        
        # Wait for deployment to be ready
        Write-Status "Waiting for deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s "deployment/$AppName"
        
        if ($LASTEXITCODE -ne 0) {
            throw "Deployment not ready within timeout"
        }
        
        Write-Status "Application deployed successfully!"
    }
    catch {
        Write-Error "Failed to deploy application: $($_.Exception.Message)"
        throw
    }
}

function Get-ApplicationInfo {
    Write-Status "Getting application information..."
    
    Write-Host ""
    Write-Host "=== CLUSTER INFORMATION ===" -ForegroundColor Cyan
    kubectl cluster-info
    
    Write-Host ""
    Write-Host "=== NODES ===" -ForegroundColor Cyan
    kubectl get nodes -o wide
    
    Write-Host ""
    Write-Host "=== PODS ===" -ForegroundColor Cyan
    kubectl get pods -o wide
    
    Write-Host ""
    Write-Host "=== SERVICES ===" -ForegroundColor Cyan
    kubectl get services -o wide
    
    Write-Host ""
    Write-Host "=== DEPLOYMENTS ===" -ForegroundColor Cyan
    kubectl get deployments -o wide
    
    Write-Host ""
    Write-Status "Application URLs:"
    Write-Host "- Streamlit App: http://localhost:8501"
    Write-Host "- API (if available): http://localhost:8080"
    Write-Host "- Registry: http://localhost:5000"
    
    Write-Host ""
    Write-Status "Useful commands:"
    Write-Host "- View logs: kubectl logs -l app=$AppName -f"
    Write-Host "- Scale app: kubectl scale deployment $AppName --replicas=3"
    Write-Host "- Delete cluster: k3d cluster delete $ClusterName"
    Write-Host "- Access cluster: `$env:KUBECONFIG = k3d kubeconfig write $ClusterName"
}

function Main {
    Write-Status "Starting k3d deployment for Vietnamese ID Card OCR Application 🚀"
    
    # Check prerequisites
    if (-not (Test-K3d)) {
        exit 1
    }
    
    if (-not (Test-Docker)) {
        exit 1
    }
    
    # Ask for confirmation unless Force is specified
    if (-not $Force) {
        $confirmation = Read-Host "This will delete existing cluster '$ClusterName'. Continue? (y/N)"
        if ($confirmation -notmatch '^[Yy]$') {
            Write-Warning "Deployment cancelled."
            exit 0
        }
    }
    
    try {
        Remove-ExistingResources
        New-K3dCluster
        Build-AndPushImage
        Deploy-Application
        Get-ApplicationInfo
        
        Write-Status "Deployment completed successfully! 🎉"
    }
    catch {
        Write-Error "Deployment failed: $($_.Exception.Message)"
        exit 1
    }
}

# Run main function
Main
