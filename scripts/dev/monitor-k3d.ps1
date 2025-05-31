# K3D Monitoring and Debug Script
# This script provides monitoring and debugging capabilities for the k3d deployment

param(
    [string]$Action = "status",
    [string]$ClusterName = "vnidcard-cluster",
    [string]$AppName = "vnidcard-app",
    [string]$Namespace = "default",
    [switch]$Follow,
    [int]$Tail = 100
)

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

function Show-ClusterInfo {
    Write-Host "=== CLUSTER INFORMATION ===" -ForegroundColor Cyan
    try {
        kubectl cluster-info
        Write-Host ""
        
        Write-Host "=== CLUSTER VERSION ===" -ForegroundColor Cyan
        kubectl version --short
        Write-Host ""
        
        Write-Host "=== NODES ===" -ForegroundColor Cyan
        kubectl get nodes -o wide
        Write-Host ""
    }
    catch {
        Write-Error "Failed to get cluster info: $($_.Exception.Message)"
    }
}

function Show-ApplicationStatus {
    Write-Host "=== APPLICATION STATUS ===" -ForegroundColor Cyan
    
    try {
        Write-Host "--- Namespaces ---" -ForegroundColor Yellow
        kubectl get namespaces
        Write-Host ""
        
        Write-Host "--- Deployments ---" -ForegroundColor Yellow
        kubectl get deployments -o wide
        Write-Host ""
        
        Write-Host "--- Pods ---" -ForegroundColor Yellow
        kubectl get pods -o wide
        Write-Host ""
        
        Write-Host "--- Services ---" -ForegroundColor Yellow
        kubectl get services -o wide
        Write-Host ""
        
        Write-Host "--- Ingress ---" -ForegroundColor Yellow
        kubectl get ingress 2>$null
        Write-Host ""
        
        Write-Host "--- HPA ---" -ForegroundColor Yellow
        kubectl get hpa 2>$null
        Write-Host ""
        
        Write-Host "--- PVC ---" -ForegroundColor Yellow
        kubectl get pvc
        Write-Host ""
    }
    catch {
        Write-Error "Failed to get application status: $($_.Exception.Message)"
    }
}

function Show-ResourceUsage {
    Write-Host "=== RESOURCE USAGE ===" -ForegroundColor Cyan
    
    try {
        Write-Host "--- Node Resource Usage ---" -ForegroundColor Yellow
        kubectl top nodes 2>$null
        Write-Host ""
        
        Write-Host "--- Pod Resource Usage ---" -ForegroundColor Yellow
        kubectl top pods 2>$null
        Write-Host ""
    }
    catch {
        Write-Warning "Resource usage metrics not available (metrics-server may not be installed)"
    }
}

function Show-Events {
    Write-Host "=== RECENT EVENTS ===" -ForegroundColor Cyan
    
    try {
        kubectl get events --sort-by=.metadata.creationTimestamp --field-selector type!=Normal | tail -20
        Write-Host ""
    }
    catch {
        Write-Error "Failed to get events: $($_.Exception.Message)"
    }
}

function Show-Logs {
    param(
        [string]$PodName = "",
        [switch]$Follow,
        [int]$Tail = 100
    )
    
    Write-Host "=== APPLICATION LOGS ===" -ForegroundColor Cyan
    
    try {
        if ($PodName) {
            Write-Status "Showing logs for pod: $PodName"
            if ($Follow) {
                kubectl logs $PodName -f --tail=$Tail
            } else {
                kubectl logs $PodName --tail=$Tail
            }
        } else {
            Write-Status "Showing logs for app: $AppName"
            if ($Follow) {
                kubectl logs -l app=$AppName -f --tail=$Tail
            } else {
                kubectl logs -l app=$AppName --tail=$Tail
            }
        }
    }
    catch {
        Write-Error "Failed to get logs: $($_.Exception.Message)"
    }
}

function Show-PodDetails {
    param([string]$PodName)
    
    Write-Host "=== POD DETAILS ===" -ForegroundColor Cyan
    
    try {
        if (-not $PodName) {
            $pods = kubectl get pods -l app=$AppName -o jsonpath='{.items[*].metadata.name}' 2>$null
            if ($pods) {
                $PodName = $pods.Split(' ')[0]
            } else {
                Write-Error "No pods found for app: $AppName"
                return
            }
        }
        
        Write-Status "Pod: $PodName"
        Write-Host ""
        
        Write-Host "--- Pod Description ---" -ForegroundColor Yellow
        kubectl describe pod $PodName
        Write-Host ""
        
        Write-Host "--- Pod YAML ---" -ForegroundColor Yellow
        kubectl get pod $PodName -o yaml
        Write-Host ""
    }
    catch {
        Write-Error "Failed to get pod details: $($_.Exception.Message)"
    }
}

function Test-Connectivity {
    Write-Host "=== CONNECTIVITY TEST ===" -ForegroundColor Cyan
    
    $urls = @(
        @{url="http://localhost:8501"; description="Streamlit UI"},
        @{url="http://localhost:8080"; description="API Endpoint"},
        @{url="http://localhost:5000"; description="Registry"}
    )
    
    foreach ($test in $urls) {
        try {
            $response = Invoke-WebRequest -Uri $test.url -Method Head -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Status "✓ $($test.description): $($test.url)"
            } else {
                Write-Warning "⚠ $($test.description): $($test.url) (Status: $($response.StatusCode))"
            }
        }
        catch {
            Write-Error "✗ $($test.description): $($test.url) (Error: $($_.Exception.Message))"
        }
    }
    Write-Host ""
}

function Show-Debug {
    Write-Host "=== DEBUG INFORMATION ===" -ForegroundColor Cyan
    
    try {
        Write-Host "--- K3D Cluster List ---" -ForegroundColor Yellow
        k3d cluster list
        Write-Host ""
        
        Write-Host "--- K3D Registry List ---" -ForegroundColor Yellow
        k3d registry list
        Write-Host ""
        
        Write-Host "--- Docker Containers ---" -ForegroundColor Yellow
        docker ps --filter "name=k3d"
        Write-Host ""
        
        Write-Host "--- Docker Networks ---" -ForegroundColor Yellow
        docker network ls | grep k3d
        Write-Host ""
        
        Write-Host "--- Docker Images ---" -ForegroundColor Yellow
        docker images | grep -E "(vnidcard|k3d)"
        Write-Host ""
    }
    catch {
        Write-Error "Failed to get debug information: $($_.Exception.Message)"
    }
}

function Restart-Application {
    Write-Host "=== RESTARTING APPLICATION ===" -ForegroundColor Cyan
    
    try {
        Write-Status "Restarting deployment: $AppName"
        kubectl rollout restart deployment/$AppName
        
        Write-Status "Waiting for rollout to complete..."
        kubectl rollout status deployment/$AppName --timeout=300s
        
        Write-Status "Application restarted successfully"
    }
    catch {
        Write-Error "Failed to restart application: $($_.Exception.Message)"
    }
}

function Scale-Application {
    param([int]$Replicas)
    
    Write-Host "=== SCALING APPLICATION ===" -ForegroundColor Cyan
    
    try {
        Write-Status "Scaling deployment $AppName to $Replicas replicas"
        kubectl scale deployment $AppName --replicas=$Replicas
        
        Write-Status "Waiting for scaling to complete..."
        kubectl rollout status deployment/$AppName --timeout=300s
        
        Write-Status "Application scaled successfully"
    }
    catch {
        Write-Error "Failed to scale application: $($_.Exception.Message)"
    }
}

function Get-Shell {
    Write-Host "=== GETTING SHELL ACCESS ===" -ForegroundColor Cyan
    
    try {
        $podName = kubectl get pods -l app=$AppName -o jsonpath='{.items[0].metadata.name}' 2>$null
        
        if (-not $podName) {
            Write-Error "No pods found for app: $AppName"
            return
        }
        
        Write-Status "Connecting to pod: $podName"
        kubectl exec -it $podName -- /bin/bash
    }
    catch {
        Write-Error "Failed to get shell access: $($_.Exception.Message)"
    }
}

function Port-Forward {
    param(
        [string]$LocalPort = "8501",
        [string]$RemotePort = "8501"
    )
    
    Write-Host "=== PORT FORWARDING ===" -ForegroundColor Cyan
    
    try {
        $podName = kubectl get pods -l app=$AppName -o jsonpath='{.items[0].metadata.name}' 2>$null
        
        if (-not $podName) {
            Write-Error "No pods found for app: $AppName"
            return
        }
        
        Write-Status "Port forwarding from localhost:$LocalPort to $podName:$RemotePort"
        Write-Status "Press Ctrl+C to stop port forwarding"
        kubectl port-forward $podName "$LocalPort:$RemotePort"
    }
    catch {
        Write-Error "Failed to setup port forwarding: $($_.Exception.Message)"
    }
}

function Show-Help {
    Write-Host "K3D Monitoring and Debug Script" -ForegroundColor Cyan
    Write-Host "===============================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\monitor-k3d.ps1 -Action <action> [options]"
    Write-Host ""
    Write-Host "Actions:"
    Write-Host "  status          Show complete status (default)"
    Write-Host "  cluster         Show cluster information"
    Write-Host "  app             Show application status"
    Write-Host "  resources       Show resource usage"
    Write-Host "  events          Show recent events"
    Write-Host "  logs            Show application logs"
    Write-Host "  debug           Show debug information"
    Write-Host "  test            Test connectivity"
    Write-Host "  restart         Restart application"
    Write-Host "  shell           Get shell access to pod"
    Write-Host "  port-forward    Setup port forwarding"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -ClusterName    K3D cluster name (default: vnidcard-cluster)"
    Write-Host "  -AppName        Application name (default: vnidcard-app)"
    Write-Host "  -Namespace      Kubernetes namespace (default: default)"
    Write-Host "  -Follow         Follow logs in real-time"
    Write-Host "  -Tail           Number of log lines to show (default: 100)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\monitor-k3d.ps1                           # Show status"
    Write-Host "  .\monitor-k3d.ps1 -Action logs -Follow      # Follow logs"
    Write-Host "  .\monitor-k3d.ps1 -Action restart           # Restart app"
    Write-Host "  .\monitor-k3d.ps1 -Action shell             # Get shell"
    Write-Host ""
}

function Main {
    switch ($Action.ToLower()) {
        "status" {
            Show-ClusterInfo
            Show-ApplicationStatus
            Show-ResourceUsage
            Test-Connectivity
        }
        "cluster" {
            Show-ClusterInfo
        }
        "app" {
            Show-ApplicationStatus
        }
        "resources" {
            Show-ResourceUsage
        }
        "events" {
            Show-Events
        }
        "logs" {
            Show-Logs -Follow:$Follow -Tail $Tail
        }
        "debug" {
            Show-Debug
        }
        "test" {
            Test-Connectivity
        }
        "restart" {
            Restart-Application
        }
        "shell" {
            Get-Shell
        }
        "port-forward" {
            Port-Forward
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "Unknown action: $Action"
            Show-Help
        }
    }
}

# Run main function
Main
