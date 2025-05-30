# Quick Start Script for Vietnamese ID Card OCR K3D Deployment
# This script provides a simple one-command deployment

param(
    [switch]$Quick,
    [switch]$Clean,
    [switch]$Status,
    [switch]$Help
)

function Show-Help {
    Write-Host "Vietnamese ID Card OCR - Quick Start" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\quick-start.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Quick     Quick deployment (check deps + deploy)"
    Write-Host "  -Clean     Clean up everything"
    Write-Host "  -Status    Show current status"
    Write-Host "  -Help      Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\quick-start.ps1 -Quick      # Quick deployment"
    Write-Host "  .\quick-start.ps1 -Status     # Check status"
    Write-Host "  .\quick-start.ps1 -Clean      # Clean up"
    Write-Host ""
}

function Quick-Deploy {
    Write-Host "🚀 Vietnamese ID Card OCR - Quick Deployment" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host ""
    
    # Check prerequisites
    Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Yellow
    .\check-prerequisites.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Prerequisites check failed. Please install missing components." -ForegroundColor Red
        return
    }
    
    # Deploy
    Write-Host ""
    Write-Host "Step 2: Deploying to k3d..." -ForegroundColor Yellow
    .\deploy-k3d.ps1 -Force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Access your application:" -ForegroundColor Cyan
        Write-Host "   Streamlit UI: http://localhost:8501" -ForegroundColor White
        Write-Host "   API Endpoint: http://localhost:8080" -ForegroundColor White
        Write-Host ""
        Write-Host "📊 Monitor your deployment:" -ForegroundColor Cyan
        Write-Host "   .\monitor-k3d.ps1" -ForegroundColor White
        Write-Host "   .\monitor-k3d.ps1 -Action logs -Follow" -ForegroundColor White
    } else {
        Write-Host "❌ Deployment failed. Check the logs above." -ForegroundColor Red
    }
}

function Clean-All {
    Write-Host "🧹 Cleaning up k3d deployment..." -ForegroundColor Yellow
    
    # Use makefile for clean operation
    if (Test-Path "Makefile.k3d") {
        make -f Makefile.k3d clean
    } else {
        # Fallback to manual cleanup
        k3d cluster delete vnidcard-cluster 2>$null
        k3d registry delete vnidcard-registry 2>$null
        docker system prune -f
    }
    
    Write-Host "✅ Cleanup completed!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "📊 Current Status" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    
    .\monitor-k3d.ps1 -Action status
}

# Main execution
if ($Help) {
    Show-Help
} elseif ($Quick) {
    Quick-Deploy
} elseif ($Clean) {
    Clean-All
} elseif ($Status) {
    Show-Status
} else {
    Show-Help
}
