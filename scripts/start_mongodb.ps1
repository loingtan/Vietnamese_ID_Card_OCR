#!/usr/bin/env pwsh
# Script to start MongoDB container for development

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$dockerComposePath = Join-Path $scriptPath "..\deployment\docker"

# Change to docker-compose directory
Set-Location -Path $dockerComposePath

# Start only the MongoDB container
Write-Host "Starting MongoDB container for development..."
docker-compose up -d mongodb

# Check if container is running
Start-Sleep -Seconds 3
$isRunning = docker ps --filter "name=vnidcard-mongodb" --format "{{.Names}}"

if ($isRunning -eq "vnidcard-mongodb") {
    Write-Host "MongoDB is running successfully!"
    Write-Host "Connection URL: mongodb://localhost:27017"
    Write-Host "Database Name: vnid_card_ocr"
} else {
    Write-Host "Failed to start MongoDB container. Check docker logs for details." -ForegroundColor Red
}

# Return to original directory
Set-Location -Path $scriptPath
