#!/usr/bin/env pwsh
# Vietnamese ID Card OCR - Quick Deploy Script
# Choose your deployment mode and get started quickly

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("development", "production", "api-only", "monitoring", "k3d", "help")]
    [string]$Mode = "help"
)

$ErrorActionPreference = "Stop"

function Show-Header {
    Write-Host ""
    Write-Host "🚀 Vietnamese ID Card OCR - Quick Deploy" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Host "Available deployment modes:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📱 development  " -ForegroundColor Green -NoNewline
    Write-Host "- Full stack with Streamlit UI (port 8501)"
    Write-Host "🏭 production   " -ForegroundColor Green -NoNewline  
    Write-Host "- API-only mode for production (no Streamlit)"
    Write-Host "⚡ api-only     " -ForegroundColor Green -NoNewline
    Write-Host "- Just API and database services"
    Write-Host "📊 monitoring   " -ForegroundColor Green -NoNewline
    Write-Host "- Full monitoring stack (Prometheus, Grafana, etc.)"
    Write-Host "☸️  k3d         " -ForegroundColor Green -NoNewline
    Write-Host "- K3D/Kubernetes optimized production deployment"
    Write-Host ""
    Write-Host "Usage Examples:" -ForegroundColor Cyan
    Write-Host "  .\quick-deploy.ps1 development   # Start dev environment"
    Write-Host "  .\quick-deploy.ps1 production    # Start production API"
    Write-Host "  .\quick-deploy.ps1 monitoring    # Start monitoring stack"
    Write-Host "  .\quick-deploy.ps1 k3d           # Start K3D deployment"
    Write-Host ""
}

function Deploy-Development {
    Write-Host "🔧 Starting Development Environment..." -ForegroundColor Yellow
    Write-Host "   • API Server (port 8080)" -ForegroundColor Gray
    Write-Host "   • Streamlit UI (port 8501)" -ForegroundColor Gray  
    Write-Host "   • Metrics (port 8000)" -ForegroundColor Gray
    Write-Host "   • MongoDB (port 27017)" -ForegroundColor Gray
    Write-Host ""
    
    Set-Location "$PSScriptRoot\docker"
    docker-compose up -d
    
    Write-Host "✅ Development environment started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "   • Streamlit UI: http://localhost:8501" -ForegroundColor Blue
    Write-Host "   • API Docs: http://localhost:8080/docs" -ForegroundColor Blue
    Write-Host "   • Health Check: http://localhost:8080/health" -ForegroundColor Blue
    Write-Host "   • Metrics: http://localhost:8000/metrics" -ForegroundColor Blue
}

function Deploy-Production {
    Write-Host "🏭 Starting Production Environment (API-only)..." -ForegroundColor Yellow
    Write-Host "   • API Server (port 8080)" -ForegroundColor Gray
    Write-Host "   • Metrics (port 8000)" -ForegroundColor Gray
    Write-Host "   • MongoDB (port 27017)" -ForegroundColor Gray
    Write-Host "   • Streamlit: DISABLED" -ForegroundColor Red
    Write-Host ""
    
    Set-Location "$PSScriptRoot\docker"
    docker-compose -f docker-compose.yml  up -d
    
    Write-Host "✅ Production environment started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "   • API Docs: http://localhost:8080/docs" -ForegroundColor Blue
    Write-Host "   • Health Check: http://localhost:8080/health" -ForegroundColor Blue
    Write-Host "   • Metrics: http://localhost:8000/metrics" -ForegroundColor Blue
    Write-Host ""
    Write-Host "🔒 Security: Streamlit UI disabled for production" -ForegroundColor Green
}

function Deploy-ApiOnly {
    Write-Host "⚡ Starting API-Only Services..." -ForegroundColor Yellow
    Write-Host "   • API Server (port 8080)" -ForegroundColor Gray
    Write-Host "   • MongoDB (port 27017)" -ForegroundColor Gray
    Write-Host ""
    
    Set-Location "$PSScriptRoot\docker"
    docker-compose --profile api up -d
    
    Write-Host "✅ API services started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "   • API Docs: http://localhost:8080/docs" -ForegroundColor Blue
    Write-Host "   • Health Check: http://localhost:8080/health" -ForegroundColor Blue
}

function Deploy-Monitoring {
    Write-Host "📊 Starting Monitoring Stack..." -ForegroundColor Yellow
    Write-Host "   • Prometheus (port 9090)" -ForegroundColor Gray
    Write-Host "   • Grafana (port 3000)" -ForegroundColor Gray
    Write-Host "   • Loki (port 3100)" -ForegroundColor Gray
    Write-Host "   • AlertManager (port 9093)" -ForegroundColor Gray
    Write-Host "   • Plus: Node Exporter, cAdvisor, Fluent Bit" -ForegroundColor Gray
    Write-Host ""
    
    Set-Location "$PSScriptRoot\docker"
    docker-compose --profile monitoring up -d
    
    Write-Host "✅ Monitoring stack started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "   • Prometheus: http://localhost:9090" -ForegroundColor Blue
    Write-Host "   • Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Blue
    Write-Host "   • AlertManager: http://localhost:9093" -ForegroundColor Blue
}

function Deploy-K3D {
    Write-Host "☸️  Starting K3D Production Deployment..." -ForegroundColor Yellow
    Write-Host "   • K3D optimized configuration" -ForegroundColor Gray
    Write-Host "   • Production API-only mode" -ForegroundColor Gray
    Write-Host "   • Container registry ready" -ForegroundColor Gray
    Write-Host ""
    
    Set-Location "$PSScriptRoot\docker"
    docker-compose -f docker-compose.yml -f docker-compose.k3d.yml up -d
    
    Write-Host "✅ K3D deployment started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "   • API: http://localhost:8080" -ForegroundColor Blue
    Write-Host "   • Metrics: http://localhost:8000" -ForegroundColor Blue
    Write-Host ""
    Write-Host "☸️  For K3D cluster deployment:" -ForegroundColor Cyan
    Write-Host "   cd ..\k3d && k3d cluster create --config k3d-config.yaml" -ForegroundColor Gray
}

function Show-Status {
    Write-Host ""
    Write-Host "📋 Current Status:" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    
    Set-Location "$PSScriptRoot\docker"
    
    $containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-Object -Skip 1
    
    if ($containers) {
        Write-Host $containers
    } else {
        Write-Host "No containers running." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "💡 Management Commands:" -ForegroundColor Cyan
    Write-Host "   • Stop all: docker-compose down" -ForegroundColor Gray
    Write-Host "   • View logs: docker-compose logs -f" -ForegroundColor Gray
    Write-Host "   • Restart: docker-compose restart" -ForegroundColor Gray
    Write-Host ""
}

# Main execution
Show-Header

switch ($Mode) {
    "development" { Deploy-Development }
    "production"  { Deploy-Production }
    "api-only"    { Deploy-ApiOnly }
    "monitoring"  { Deploy-Monitoring }
    "k3d"         { Deploy-K3D }
    "help"        { Show-Help }
    default       { Show-Help }
}

if ($Mode -ne "help") {
    Start-Sleep 3
    Show-Status
}
