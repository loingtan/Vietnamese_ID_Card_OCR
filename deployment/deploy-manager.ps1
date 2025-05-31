#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Vietnamese ID Card OCR - Deployment Manager
    
.DESCRIPTION
    This script provides easy management of the consolidated Docker deployment
    for the Vietnamese ID Card OCR application with integrated monitoring.
    
.PARAMETER Action
    The action to perform: start, stop, restart, status, logs, clean
    
.PARAMETER Profile
    The deployment profile: api, monitoring, gpu, all (default: api)
    
.PARAMETER Environment  
    The environment: production, development, k3d (default: production)
    
.EXAMPLE
    .\deploy-manager.ps1 -Action start -Profile all
    Start complete stack with monitoring
    
.EXAMPLE
    .\deploy-manager.ps1 -Action start -Profile api -Environment development
    Start API only in development mode
    
.EXAMPLE
    .\deploy-manager.ps1 -Action logs -Profile monitoring
    Show logs for monitoring services
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "clean")]
    [string]$Action,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("api", "monitoring", "gpu", "all")]
    [string]$Profile = "api",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("production", "development", "k3d")]
    [string]$Environment = "production"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Define colors for output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $Blue)
    Write-Host "${Color}${Message}${Reset}"
}

function Write-Status { param([string]$Message) Write-ColorOutput "ℹ️ $Message" $Blue }
function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" $Green }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️ $Message" $Yellow }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" $Red }

# Check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed or not in PATH"
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose is not installed or not in PATH"
        exit 1
    }
    
    # Test Docker daemon
    try {
        docker info | Out-Null
    } catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
    
    Write-Success "All prerequisites are available"
}

# Get compose files based on environment
function Get-ComposeFiles {
    param([string]$Env)
    
    $composeFiles = @()
    
    switch ($Env) {
        "production" {
            $composeFiles += "docker-compose.yml"
        }
        "development" {
            $composeFiles += "docker-compose.yml"
            $composeFiles += "docker-compose.override.yml"
        }
        "k3d" {
            $composeFiles += "docker-compose.k3d.yml"
        }
    }
    
    return $composeFiles
}

# Build docker-compose command
function Build-ComposeCommand {
    param([string[]]$Files, [string]$Prof, [string]$Cmd)
    
    $command = "docker-compose"
    
    # Add compose files
    foreach ($file in $Files) {
        $command += " -f $file"
    }
    
    # Add profiles
    if ($Prof -eq "all") {
        $command += " --profile monitoring"
    } elseif ($Prof -eq "monitoring") {
        $command += " --profile monitoring"
    } elseif ($Prof -eq "gpu") {
        $command += " --profile gpu --profile monitoring"
    }
    
    # Add command
    $command += " $Cmd"
    
    return $command
}

# Navigate to deployment directory
function Set-DeploymentLocation {
    $deploymentPath = Join-Path $PSScriptRoot "..\deployment\docker"
    if (-not (Test-Path $deploymentPath)) {
        Write-Error "Deployment directory not found: $deploymentPath"
        exit 1
    }
    
    Set-Location $deploymentPath
    Write-Status "Changed to deployment directory: $(Get-Location)"
}

# Start services
function Start-Services {
    Write-Status "Starting services with profile: $Profile, environment: $Environment"
    
    $composeFiles = Get-ComposeFiles $Environment
    $command = Build-ComposeCommand $composeFiles $Profile "up -d"
    
    Write-Status "Running: $command"
    Invoke-Expression $command
    
    Write-Success "Services started successfully!"
    
    # Wait and check health
    Write-Status "Waiting for services to start..."
    Start-Sleep 10
    Show-ServiceStatus
}

# Stop services
function Stop-Services {
    Write-Status "Stopping services..."
    
    $composeFiles = Get-ComposeFiles $Environment
    $command = Build-ComposeCommand $composeFiles $Profile "down"
    
    Write-Status "Running: $command"
    Invoke-Expression $command
    
    Write-Success "Services stopped successfully!"
}

# Restart services
function Restart-Services {
    Write-Status "Restarting services..."
    Stop-Services
    Start-Sleep 5
    Start-Services
}

# Show service status
function Show-ServiceStatus {
    Write-Status "Checking service status..."
    
    $composeFiles = Get-ComposeFiles $Environment
    $command = Build-ComposeCommand $composeFiles $Profile "ps"
    
    Write-Status "Running: $command"
    Invoke-Expression $command
      # Check health endpoints
    Write-Status "Checking health endpoints..."
    
    $healthChecks = @{
        "API" = "http://localhost:8080/health"
        "Metrics" = "http://localhost:8000/metrics"
    }
    
    # Add Streamlit check only for development environment
    if ($Environment -eq "development") {
        $healthChecks["Streamlit"] = "http://localhost:8501/_stcore/health"
    }
    
    # Add monitoring checks if monitoring profile is enabled
    if ($Profile -eq "all" -or $Profile -eq "monitoring") {
        $healthChecks["Prometheus"] = "http://localhost:9090/-/ready"
        $healthChecks["Grafana"] = "http://localhost:3000/api/health"
        $healthChecks["Loki"] = "http://localhost:3100/ready"
    }
    
    foreach ($service in $healthChecks.Keys) {
        $url = $healthChecks[$service]
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Success "$service is healthy"
            } else {
                Write-Warning "$service returned status: $($response.StatusCode)"
            }
        } catch {
            Write-Warning "$service is not responding"
        }
    }
}

# Show logs
function Show-Logs {
    Write-Status "Showing logs for profile: $Profile"
    
    $composeFiles = Get-ComposeFiles $Environment
    $command = Build-ComposeCommand $composeFiles $Profile "logs -f --tail=100"
    
    Write-Status "Running: $command"
    Write-Status "Press Ctrl+C to stop following logs..."
    Invoke-Expression $command
}

# Clean up
function Clean-Environment {
    Write-Warning "This will remove all containers, volumes, and images!"
    $confirm = Read-Host "Are you sure? (y/N)"
    
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Write-Status "Cleaning up environment..."
        
        $composeFiles = Get-ComposeFiles $Environment
        $command = Build-ComposeCommand $composeFiles $Profile "down -v --remove-orphans"
        
        Write-Status "Running: $command"
        Invoke-Expression $command
        
        Write-Status "Pruning Docker system..."
        docker system prune -f
        
        Write-Success "Environment cleaned successfully!"
    } else {
        Write-Status "Cleanup cancelled"
    }
}

# Display access information
function Show-AccessInfo {
    Write-Success "🎉 Deployment Manager Complete!"
    Write-Host ""
    Write-ColorOutput "🚀 Access your services:" $Green
    Write-Host "   • API Application:      http://localhost:8080"
    Write-Host "   • API Metrics:          http://localhost:8000"
    
    # Show Streamlit only in development environment
    if ($Environment -eq "development") {
        Write-Host "   • Streamlit UI:         http://localhost:8501"
    }
    Write-Host ""
    
    if ($Profile -eq "all" -or $Profile -eq "monitoring") {
        Write-ColorOutput "📊 Monitoring Services:" $Green
        Write-Host "   • Grafana Dashboard:    http://localhost:3000"
        Write-Host "     - Username: admin"
        Write-Host "     - Password: vnidcard123"
        Write-Host ""
        Write-Host "   • Prometheus:           http://localhost:9090"
        Write-Host "   • Alertmanager:         http://localhost:9093"
        Write-Host "   • Loki:                 http://localhost:3100"
        Write-Host ""
    }
    
    Write-ColorOutput "📚 API Documentation:" $Blue
    Write-Host "   • Swagger UI:           http://localhost:8080/docs"
    Write-Host "   • ReDoc:                http://localhost:8080/redoc"
    Write-Host "   • Health Check:         http://localhost:8080/health"
    Write-Host ""
    
    Write-ColorOutput "💡 Management Commands:" $Blue
    Write-Host "   .\deploy-manager.ps1 -Action status    # Check status"
    Write-Host "   .\deploy-manager.ps1 -Action logs      # View logs"
    Write-Host "   .\deploy-manager.ps1 -Action stop      # Stop services"
    Write-Host "   .\deploy-manager.ps1 -Action clean     # Clean up"
    Write-Host ""
}

# Main execution
function Main {
    Write-ColorOutput "🐋 Vietnamese ID Card OCR - Deployment Manager" $Blue
    Write-Host ""
    
    Test-Prerequisites
    Set-DeploymentLocation
    
    switch ($Action) {
        "start" {
            Start-Services
            Show-AccessInfo
        }
        "stop" {
            Stop-Services
        }
        "restart" {
            Restart-Services
            Show-AccessInfo
        }
        "status" {
            Show-ServiceStatus
        }
        "logs" {
            Show-Logs
        }
        "clean" {
            Clean-Environment
        }
    }
}

# Execute main function
try {
    Main
} catch {
    Write-Error "An error occurred: $($_.Exception.Message)"
    exit 1
}
