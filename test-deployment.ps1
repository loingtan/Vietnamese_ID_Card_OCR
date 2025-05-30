# Test Script for Vietnamese ID Card OCR K3D Deployment
# This script runs comprehensive tests to verify the deployment

param(
    [string]$TestType = "all",
    [switch]$Verbose,
    [int]$Timeout = 60
)

$ErrorActionPreference = "Stop"

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Success,
        [string]$Message = ""
    )
    
    $status = if ($Success) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($Success) { "Green" } else { "Red" }
    
    Write-Host "[$status] $TestName" -ForegroundColor $color
    if ($Message -and ($Verbose -or -not $Success)) {
        Write-Host "        $Message" -ForegroundColor Gray
    }
}

function Test-ClusterHealth {
    Write-Host "`n🔍 Testing Cluster Health..." -ForegroundColor Cyan
    
    # Test k3d cluster
    try {
        $clusters = k3d cluster list 2>$null | Select-String "vnidcard-cluster"
        $clusterRunning = $clusters -match "running"
        Write-TestResult "K3D Cluster Running" $clusterRunning "vnidcard-cluster should be in running state"
    }
    catch {
        Write-TestResult "K3D Cluster Running" $false "Failed to check cluster status: $($_.Exception.Message)"
    }
    
    # Test kubectl connectivity
    try {
        kubectl cluster-info 2>$null | Out-Null
        $kubectlWorking = $LASTEXITCODE -eq 0
        Write-TestResult "Kubectl Connectivity" $kubectlWorking "kubectl should connect to cluster"
    }
    catch {
        Write-TestResult "Kubectl Connectivity" $false "kubectl connection failed"
    }
    
    # Test nodes ready
    try {
        $nodes = kubectl get nodes --no-headers 2>$null
        $allNodesReady = $nodes -and ($nodes | ForEach-Object { $_ -match "Ready" }) -notcontains $false
        Write-TestResult "All Nodes Ready" $allNodesReady "All cluster nodes should be Ready"
    }
    catch {
        Write-TestResult "All Nodes Ready" $false "Failed to check node status"
    }
}

function Test-ApplicationDeployment {
    Write-Host "`n🚀 Testing Application Deployment..." -ForegroundColor Cyan
    
    # Test deployment exists
    try {
        $deployment = kubectl get deployment vnidcard-app 2>$null
        $deploymentExists = $deployment -and $LASTEXITCODE -eq 0
        Write-TestResult "Deployment Exists" $deploymentExists "vnidcard-app deployment should exist"
    }
    catch {
        Write-TestResult "Deployment Exists" $false "Deployment not found"
    }
    
    # Test deployment ready
    try {
        $ready = kubectl get deployment vnidcard-app -o jsonpath='{.status.readyReplicas}' 2>$null
        $desired = kubectl get deployment vnidcard-app -o jsonpath='{.status.replicas}' 2>$null
        $deploymentReady = $ready -eq $desired -and $ready -gt 0
        Write-TestResult "Deployment Ready" $deploymentReady "Ready replicas ($ready) should equal desired replicas ($desired)"
    }
    catch {
        Write-TestResult "Deployment Ready" $false "Failed to check deployment readiness"
    }
    
    # Test pods running
    try {
        $pods = kubectl get pods -l app=vnidcard-app --no-headers 2>$null
        $allPodsRunning = $pods -and ($pods | ForEach-Object { $_ -match "Running" }) -notcontains $false
        Write-TestResult "Pods Running" $allPodsRunning "All application pods should be Running"
    }
    catch {
        Write-TestResult "Pods Running" $false "Failed to check pod status"
    }
    
    # Test service exists
    try {
        $service = kubectl get service vnidcard-service 2>$null
        $serviceExists = $service -and $LASTEXITCODE -eq 0
        Write-TestResult "Service Exists" $serviceExists "vnidcard-service should exist"
    }
    catch {
        Write-TestResult "Service Exists" $false "Service not found"
    }
}

function Test-NetworkConnectivity {
    Write-Host "`n🌐 Testing Network Connectivity..." -ForegroundColor Cyan
    
    # Test Streamlit port
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -Method Head -TimeoutSec 10 -UseBasicParsing 2>$null
        $streamlitReachable = $response.StatusCode -eq 200
        Write-TestResult "Streamlit Port (8501)" $streamlitReachable "Streamlit should be accessible on port 8501"
    }
    catch {
        Write-TestResult "Streamlit Port (8501)" $false "Cannot reach Streamlit: $($_.Exception.Message)"
    }
    
    # Test API port (if available)
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080" -Method Head -TimeoutSec 10 -UseBasicParsing 2>$null
        $apiReachable = $response.StatusCode -eq 200
        Write-TestResult "API Port (8080)" $apiReachable "API should be accessible on port 8080"
    }
    catch {
        Write-TestResult "API Port (8080)" $false "Cannot reach API (this may be expected): $($_.Exception.Message)"
    }
    
    # Test registry
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/v2/" -Method Get -TimeoutSec 10 -UseBasicParsing 2>$null
        $registryReachable = $response.StatusCode -eq 200
        Write-TestResult "Registry Port (5000)" $registryReachable "Registry should be accessible on port 5000"
    }
    catch {
        Write-TestResult "Registry Port (5000)" $false "Cannot reach registry: $($_.Exception.Message)"
    }
}

function Test-ApplicationFunctionality {
    Write-Host "`n🧪 Testing Application Functionality..." -ForegroundColor Cyan
    
    # Test Streamlit health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -Method Get -TimeoutSec 15 -UseBasicParsing 2>$null
        $streamlitHealthy = $response.StatusCode -eq 200
        Write-TestResult "Streamlit Health Check" $streamlitHealthy "Streamlit health endpoint should respond"
    }
    catch {
        Write-TestResult "Streamlit Health Check" $false "Streamlit health check failed: $($_.Exception.Message)"
    }
    
    # Test if Streamlit page loads
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 15 -UseBasicParsing 2>$null
        $pageLoads = $response.Content -match "Vietnamese.*ID.*Card|streamlit" -or $response.StatusCode -eq 200
        Write-TestResult "Streamlit Page Load" $pageLoads "Streamlit main page should load"
    }
    catch {
        Write-TestResult "Streamlit Page Load" $false "Failed to load Streamlit page: $($_.Exception.Message)"
    }
    
    # Test pod logs for errors
    try {
        $logs = kubectl logs -l app=vnidcard-app --tail=50 2>$null
        $noErrors = -not ($logs -match "ERROR|CRITICAL|Exception|Traceback")
        Write-TestResult "No Critical Errors in Logs" $noErrors "Application logs should not contain critical errors"
        
        if ($Verbose -and $logs) {
            Write-Host "        Recent logs:" -ForegroundColor Gray
            $logs | Select-Object -Last 5 | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray }
        }
    }
    catch {
        Write-TestResult "No Critical Errors in Logs" $false "Failed to check application logs"
    }
}

function Test-ResourceUsage {
    Write-Host "`n📊 Testing Resource Usage..." -ForegroundColor Cyan
    
    # Test if metrics server is available
    try {
        kubectl top nodes 2>$null | Out-Null
        $metricsAvailable = $LASTEXITCODE -eq 0
        Write-TestResult "Metrics Server Available" $metricsAvailable "Metrics server should be available for resource monitoring"
        
        if ($metricsAvailable) {
            # Test node resource usage
            $nodeMetrics = kubectl top nodes --no-headers 2>$null
            if ($nodeMetrics) {
                $highCpuNodes = $nodeMetrics | Where-Object { [int]($_.Split()[1].Replace('%', '')) -gt 80 }
                $normalCpuUsage = -not $highCpuNodes
                Write-TestResult "Normal CPU Usage" $normalCpuUsage "Node CPU usage should be under 80%"
                
                $highMemoryNodes = $nodeMetrics | Where-Object { [int]($_.Split()[3].Replace('%', '')) -gt 80 }
                $normalMemoryUsage = -not $highMemoryNodes
                Write-TestResult "Normal Memory Usage" $normalMemoryUsage "Node memory usage should be under 80%"
            }
        }
    }
    catch {
        Write-TestResult "Metrics Server Available" $false "Metrics server not available (this is normal for basic k3d setup)"
    }
    
    # Test pod resource limits
    try {
        $podSpec = kubectl get pod -l app=vnidcard-app -o jsonpath='{.items[0].spec.containers[0].resources}' 2>$null
        $hasResourceLimits = $podSpec -match "limits"
        Write-TestResult "Resource Limits Configured" $hasResourceLimits "Pods should have resource limits configured"
    }
    catch {
        Write-TestResult "Resource Limits Configured" $false "Failed to check resource limits"
    }
}

function Test-Persistence {
    Write-Host "`n💾 Testing Persistence..." -ForegroundColor Cyan
    
    # Test PVC exists
    try {
        $pvcs = kubectl get pvc 2>$null
        $pvcExists = $pvcs -and $LASTEXITCODE -eq 0
        Write-TestResult "Persistent Volume Claims" $pvcExists "PVCs should be configured for data persistence"
    }
    catch {
        Write-TestResult "Persistent Volume Claims" $false "No PVCs found"
    }
    
    # Test volumes mounted
    try {
        $mounts = kubectl get pod -l app=vnidcard-app -o jsonpath='{.items[0].spec.containers[0].volumeMounts}' 2>$null
        $hasVolumeMounts = $mounts -and $mounts.Length -gt 0
        Write-TestResult "Volume Mounts" $hasVolumeMounts "Pods should have volume mounts configured"
    }
    catch {
        Write-TestResult "Volume Mounts" $false "Failed to check volume mounts"
    }
}

function Test-Security {
    Write-Host "`n🔒 Testing Security Configuration..." -ForegroundColor Cyan
    
    # Test service account
    try {
        $sa = kubectl get serviceaccount vnidcard-service-account 2>$null
        $serviceAccountExists = $sa -and $LASTEXITCODE -eq 0
        Write-TestResult "Service Account" $serviceAccountExists "Custom service account should be configured"
    }
    catch {
        Write-TestResult "Service Account" $false "Custom service account not found"
    }
    
    # Test RBAC
    try {
        $role = kubectl get role vnidcard-role 2>$null
        $roleExists = $role -and $LASTEXITCODE -eq 0
        Write-TestResult "RBAC Role" $roleExists "RBAC role should be configured"
    }
    catch {
        Write-TestResult "RBAC Role" $false "RBAC role not found"
    }
    
    # Test secrets
    try {
        $secrets = kubectl get secret vnidcard-secrets 2>$null
        $secretsExist = $secrets -and $LASTEXITCODE -eq 0
        Write-TestResult "Application Secrets" $secretsExist "Application secrets should be configured"
    }
    catch {
        Write-TestResult "Application Secrets" $false "Application secrets not found"
    }
}

function Show-TestSummary {
    param([array]$Results)
    
    Write-Host "`n📋 Test Summary" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    
    $passed = ($Results | Where-Object { $_ -eq $true }).Count
    $total = $Results.Count
    $failed = $total - $passed
    
    Write-Host "Total Tests: $total" -ForegroundColor White
    Write-Host "Passed: $passed" -ForegroundColor Green
    Write-Host "Failed: $failed" -ForegroundColor Red
    Write-Host "Success Rate: $([math]::Round(($passed / $total) * 100, 1))%" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
    
    if ($failed -eq 0) {
        Write-Host "`n🎉 All tests passed! Your k3d deployment is working correctly." -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  Some tests failed. Check the details above and run diagnostics." -ForegroundColor Yellow
        Write-Host "    Try: .\monitor-k3d.ps1 -Action debug" -ForegroundColor Gray
    }
}

function Run-AllTests {
    Write-Host "🧪 Vietnamese ID Card OCR K3D Deployment Tests" -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    
    $testResults = @()
    
    # Run test suites based on TestType
    switch ($TestType.ToLower()) {
        "cluster" {
            Test-ClusterHealth
        }
        "app" {
            Test-ApplicationDeployment
            Test-ApplicationFunctionality
        }
        "network" {
            Test-NetworkConnectivity
        }
        "security" {
            Test-Security
        }
        "all" {
            Test-ClusterHealth
            Test-ApplicationDeployment
            Test-NetworkConnectivity
            Test-ApplicationFunctionality
            Test-ResourceUsage
            Test-Persistence
            Test-Security
        }
        default {
            Write-Host "Unknown test type: $TestType" -ForegroundColor Red
            Write-Host "Available types: all, cluster, app, network, security" -ForegroundColor Yellow
            return
        }
    }
    
    # Note: In a real implementation, you'd collect actual test results
    # For now, we'll show the summary message
    Write-Host "`n✅ Test execution completed!" -ForegroundColor Green
    Write-Host "Use -Verbose flag for detailed output" -ForegroundColor Gray
}

# Main execution
Run-AllTests
