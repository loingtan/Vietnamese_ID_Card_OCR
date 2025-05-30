# Vietnamese ID Card OCR - Prerequisites Check Script
# This script checks and installs required dependencies for the monitoring system and k3d deployment

param(
    [switch]$Install,
    [switch]$Force,
    [switch]$Monitoring,
    [switch]$Verbose
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

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Test-Docker {
    Write-Status "Checking Docker..."
    
    if (-not (Test-Command "docker")) {
        Write-Error "Docker is not installed or not in PATH"
        if ($Install) {
            Write-Status "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
            Write-Status "Then restart this script"
        }
        return $false
    }
    
    try {
        $dockerVersion = docker --version
        Write-Status "Docker found: $dockerVersion"
        
        # Check if Docker is running
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is installed but not running. Please start Docker Desktop"
            return $false
        }
        
        Write-Status "Docker is running"
        return $true
    }
    catch {
        Write-Error "Failed to check Docker status"
        return $false
    }
}

function Test-K3d {
    Write-Status "Checking k3d..."
    
    if (-not (Test-Command "k3d")) {
        Write-Error "k3d is not installed or not in PATH"
        if ($Install) {
            Install-K3d
        }
        return (Test-Command "k3d")
    }
    
    try {
        $k3dVersion = k3d version
        Write-Status "k3d found: $k3dVersion"
        return $true
    }
    catch {
        Write-Error "Failed to check k3d version"
        return $false
    }
}

function Install-K3d {
    Write-Status "Installing k3d..."
    
    try {
        # Check if Chocolatey is available
        if (Test-Command "choco") {
            Write-Status "Installing k3d using Chocolatey..."
            choco install k3d -y
        }
        # Check if Scoop is available
        elseif (Test-Command "scoop") {
            Write-Status "Installing k3d using Scoop..."
            scoop install k3d
        }
        # Manual installation
        else {
            Write-Status "Installing k3d manually..."
            
            # Create temp directory
            $tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -Type Directory -Path $_ }
            
            # Download k3d
            $k3dUrl = "https://github.com/k3d-io/k3d/releases/latest/download/k3d-windows-amd64.exe"
            $k3dPath = Join-Path $tempDir "k3d.exe"
            
            Write-Status "Downloading k3d from $k3dUrl"
            Invoke-WebRequest -Uri $k3dUrl -OutFile $k3dPath
            
            # Install to a directory in PATH
            $installDir = "$env:LOCALAPPDATA\k3d"
            if (-not (Test-Path $installDir)) {
                New-Item -ItemType Directory -Path $installDir -Force
            }
            
            Copy-Item $k3dPath "$installDir\k3d.exe" -Force
            
            # Add to PATH if not already there
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if ($currentPath -notlike "*$installDir*") {
                $newPath = "$currentPath;$installDir"
                [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
                Write-Warning "Added $installDir to user PATH. Please restart PowerShell or reload your profile"
            }
            
            # Clean up
            Remove-Item $tempDir -Recurse -Force
        }
        
        Write-Status "k3d installation completed"
    }
    catch {
        Write-Error "Failed to install k3d: $($_.Exception.Message)"
        return $false
    }
}

function Test-Kubectl {
    Write-Status "Checking kubectl..."
    
    if (-not (Test-Command "kubectl")) {
        Write-Error "kubectl is not installed or not in PATH"
        if ($Install) {
            Install-Kubectl
        }
        return (Test-Command "kubectl")
    }
    
    try {
        $kubectlVersion = kubectl version --client --short 2>$null
        Write-Status "kubectl found: $kubectlVersion"
        return $true
    }
    catch {
        Write-Error "Failed to check kubectl version"
        return $false
    }
}

function Install-Kubectl {
    Write-Status "Installing kubectl..."
    
    try {
        # Check if Chocolatey is available
        if (Test-Command "choco") {
            Write-Status "Installing kubectl using Chocolatey..."
            choco install kubernetes-cli -y
        }
        # Check if Scoop is available
        elseif (Test-Command "scoop") {
            Write-Status "Installing kubectl using Scoop..."
            scoop install kubectl
        }
        # Manual installation
        else {
            Write-Status "Installing kubectl manually..."
            
            # Create temp directory
            $tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -Type Directory -Path $_ }
            
            # Get latest version
            $latestVersion = (Invoke-WebRequest -Uri "https://dl.k8s.io/release/stable.txt" -UseBasicParsing).Content.Trim()
            
            # Download kubectl
            $kubectlUrl = "https://dl.k8s.io/release/$latestVersion/bin/windows/amd64/kubectl.exe"
            $kubectlPath = Join-Path $tempDir "kubectl.exe"
            
            Write-Status "Downloading kubectl $latestVersion from $kubectlUrl"
            Invoke-WebRequest -Uri $kubectlUrl -OutFile $kubectlPath
            
            # Install to a directory in PATH
            $installDir = "$env:LOCALAPPDATA\kubectl"
            if (-not (Test-Path $installDir)) {
                New-Item -ItemType Directory -Path $installDir -Force
            }
            
            Copy-Item $kubectlPath "$installDir\kubectl.exe" -Force
            
            # Add to PATH if not already there
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if ($currentPath -notlike "*$installDir*") {
                $newPath = "$currentPath;$installDir"
                [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
                Write-Warning "Added $installDir to user PATH. Please restart PowerShell or reload your profile"
            }
            
            # Clean up
            Remove-Item $tempDir -Recurse -Force
        }
        
        Write-Status "kubectl installation completed"
    }
    catch {
        Write-Error "Failed to install kubectl: $($_.Exception.Message)"
        return $false
    }
}

function Test-SystemRequirements {
    Write-Status "Checking system requirements..."
    
    # Check available memory
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 1)
    
    if ($totalMemoryGB -lt 8) {
        Write-Warning "System has only $totalMemoryGB GB RAM. Recommended: 8GB minimum, 16GB preferred"
    } else {
        Write-Status "Memory: $totalMemoryGB GB (✓)"
    }
    
    # Check CPU cores
    $cpu = Get-WmiObject -Class Win32_Processor
    $cores = $cpu.NumberOfCores
    
    if ($cores -lt 4) {
        Write-Warning "System has only $cores CPU cores. Recommended: 4 cores minimum"
    } else {
        Write-Status "CPU Cores: $cores (✓)"
    }
    
    # Check available disk space
    $disk = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 -and $_.DeviceID -eq "C:" }
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 1)
    
    if ($freeSpaceGB -lt 10) {
        Write-Warning "Only $freeSpaceGB GB free disk space. Recommended: 10GB minimum"
    } else {
        Write-Status "Free Disk Space: $freeSpaceGB GB (✓)"
    }
}

function Test-NetworkConnectivity {
    Write-Status "Checking network connectivity..."
    
    $testUrls = @(
        "https://github.com",
        "https://registry-1.docker.io",
        "https://k3d.io"
    )
    
    foreach ($url in $testUrls) {
        try {
            $response = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Status "✓ $url"
            } else {
                Write-Warning "⚠ $url (Status: $($response.StatusCode))"
            }
        }
        catch {
            Write-Error "✗ $url (Error: $($_.Exception.Message))"
        }
    }
}

function Test-MonitoringPrerequisites {
    Write-Status "Checking monitoring system prerequisites..."
    
    $passed = 0
    $failed = 0
    $warnings = 0
    
    # Check PowerShell version
    Write-Host "PowerShell Version: " -NoNewline
    $version = $PSVersionTable.PSVersion
    if ($version.Major -ge 5 -and $version.Minor -ge 1) {
        Write-Host "✓ $version" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "✗ $version (Need ≥5.1)" -ForegroundColor Red
        $failed++
    }
    
    # Check available memory
    Write-Host "Available Memory: " -NoNewline
    $memory = Get-CimInstance Win32_OperatingSystem
    $availableGB = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
    if ($availableGB -ge 4) {
        Write-Host "✓ ${availableGB}GB" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "⚠ ${availableGB}GB (Recommended ≥4GB)" -ForegroundColor Yellow
        $warnings++
    }
    
    # Check available disk space
    Write-Host "Available Disk Space: " -NoNewline
    $drive = Get-PSDrive -Name C
    $freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
    if ($freeSpaceGB -ge 10) {
        Write-Host "✓ ${freeSpaceGB}GB" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "⚠ ${freeSpaceGB}GB (Recommended ≥10GB)" -ForegroundColor Yellow
        $warnings++
    }
    
    # Check Python installation
    Write-Host "Python: " -NoNewline
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion) {
            $version = [Version]($pythonVersion -replace "Python ", "")
            if ($version -ge [Version]"3.8.0") {
                Write-Host "✓ $version" -ForegroundColor Green
                $passed++
            } else {
                Write-Host "✗ $version (Need ≥3.8)" -ForegroundColor Red
                $failed++
            }
        } else {
            Write-Host "✗ Not installed" -ForegroundColor Red
            $failed++
        }
    } catch {
        Write-Host "✗ Not installed" -ForegroundColor Red
        $failed++
    }
    
    # Check monitoring configuration files
    $requiredFiles = @(
        "monitoring\docker-compose.monitoring.yml",
        "monitoring\prometheus\prometheus.yml",
        "monitoring\grafana\provisioning\datasources\datasources.yml",
        "src\api\fastapi_app.py"
    )
    
    Write-Host "Configuration Files: " -NoNewline
    $filesOk = $true
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            $filesOk = $false
            if ($Verbose) { Write-Warning "Missing: $file" }
        }
    }
    
    if ($filesOk) {
        Write-Host "✓ All present" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "✗ Some missing" -ForegroundColor Red
        $failed++
    }
    
    # Check port availability
    $ports = @(3000, 8080, 9090, 9093, 3100)
    Write-Host "Port Availability: " -NoNewline
    $portsOk = $true
    foreach ($port in $ports) {
        try {
            $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
            if ($connection.TcpTestSucceeded) {
                $portsOk = $false
                if ($Verbose) { Write-Warning "Port $port is in use" }
            }
        } catch {
            # If test fails, assume port is available
        }
    }
    
    if ($portsOk) {
        Write-Host "✓ All available" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "⚠ Some in use" -ForegroundColor Yellow
        $warnings++
    }
    
    return @{
        Passed = $passed
        Failed = $failed
        Warnings = $warnings
        Ready = ($failed -eq 0)
    }
}

function Show-Summary {
    Write-Host ""
    Write-Host "=== PREREQUISITES SUMMARY ===" -ForegroundColor Cyan
    Write-Host ""
    
    $dockerOk = Test-Docker
    $k3dOk = Test-K3d
    $kubectlOk = Test-Kubectl
    
    Write-Host "Docker:  " -NoNewline
    if ($dockerOk) { Write-Host "✓ Ready" -ForegroundColor Green } else { Write-Host "✗ Not Ready" -ForegroundColor Red }
    
    Write-Host "k3d:     " -NoNewline
    if ($k3dOk) { Write-Host "✓ Ready" -ForegroundColor Green } else { Write-Host "✗ Not Ready" -ForegroundColor Red }
    
    Write-Host "kubectl: " -NoNewline
    if ($kubectlOk) { Write-Host "✓ Ready" -ForegroundColor Green } else { Write-Host "✗ Not Ready" -ForegroundColor Red }
    
    Write-Host ""
    
    if ($dockerOk -and $k3dOk -and $kubectlOk) {
        Write-Status "All prerequisites are ready! You can now run the deployment."
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  .\deploy-k3d.ps1                    # Full deployment"
        Write-Host "  make -f Makefile.k3d all            # Using Makefile"
        Write-Host ""
    } else {
        Write-Warning "Some prerequisites are missing. Use -Install flag to install them automatically."
        Write-Host ""
        Write-Host "To install missing components:"
        Write-Host "  .\check-prerequisites.ps1 -Install"
        Write-Host ""
    }
}

function Show-MonitoringSummary {
    param($Results)
    
    Write-Host ""
    Write-Host "Monitoring System Summary:" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    
    Write-Host "Passed: " -NoNewline
    Write-Host $Results.Passed -ForegroundColor Green
    
    if ($Results.Warnings -gt 0) {
        Write-Host "Warnings: " -NoNewline
        Write-Host $Results.Warnings -ForegroundColor Yellow
    }
    
    if ($Results.Failed -gt 0) {
        Write-Host "Failed: " -NoNewline
        Write-Host $Results.Failed -ForegroundColor Red
    }
    
    Write-Host ""
    
    if ($Results.Ready) {
        Write-Host "🎉 Monitoring system prerequisites are ready!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Configure .env file with API keys" -ForegroundColor White
        Write-Host "  2. Run: .\monitoring\start-monitoring.bat" -ForegroundColor White
        Write-Host "  3. Access Grafana at http://localhost:3000" -ForegroundColor White
    } else {
        Write-Host "🚫 Some monitoring prerequisites are missing!" -ForegroundColor Red
        Write-Host "Please fix the failed requirements before proceeding." -ForegroundColor Red
    }
}

function Main {
    Write-Host "K3D Prerequisites Checker" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host ""
    
    Test-SystemRequirements
    Write-Host ""
    
    Test-NetworkConnectivity
    Write-Host ""
    
    if ($Install) {
        Write-Status "Installing missing prerequisites..."
        
        if (-not (Test-Docker)) {
            Write-Status "Please install Docker Desktop manually from: https://www.docker.com/products/docker-desktop"
        }
        
        if (-not (Test-K3d)) {
            # Installation handled in Test-K3d
        }
        
        if (-not (Test-Kubectl)) {
            # Installation handled in Test-Kubectl
        }
        
        Write-Host ""
    }
    
    Show-Summary
}

# Run main function
Main
