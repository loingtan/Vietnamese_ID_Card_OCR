@echo off
REM Vietnamese ID Card OCR - Monitoring Stack Startup Script (Windows)
REM This script starts the complete monitoring infrastructure on Windows

echo 🚀 Starting Vietnamese ID Card OCR Monitoring Stack...

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not available. Please install Docker Desktop with Compose.
    pause
    exit /b 1
)

echo [INFO] Docker and Docker Compose are available

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "monitoring\prometheus\data" mkdir monitoring\prometheus\data
if not exist "monitoring\grafana\data" mkdir monitoring\grafana\data
if not exist "monitoring\loki\data" mkdir monitoring\loki\data
if not exist "monitoring\alertmanager\data" mkdir monitoring\alertmanager\data

echo [SUCCESS] Directories created successfully

REM Start the monitoring stack
echo [INFO] Starting monitoring stack...
cd monitoring

REM Pull latest images
echo [INFO] Pulling latest Docker images...
docker-compose -f docker-compose.monitoring.yml pull

REM Start services
echo [INFO] Starting services...
docker-compose -f docker-compose.monitoring.yml up -d

echo [SUCCESS] Monitoring stack started successfully!

REM Wait for services to start
echo [INFO] Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Check service health
echo [INFO] Checking service health...

curl -f -s "http://localhost:9090" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Prometheus might not be ready yet (port 9090)
) else (
    echo [SUCCESS] Prometheus is running on port 9090
)

curl -f -s "http://localhost:3000" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Grafana might not be ready yet (port 3000)
) else (
    echo [SUCCESS] Grafana is running on port 3000
)

curl -f -s "http://localhost:9093" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Alertmanager might not be ready yet (port 9093)
) else (
    echo [SUCCESS] Alertmanager is running on port 9093
)

curl -f -s "http://localhost:3100" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Loki might not be ready yet (port 3100)
) else (
    echo [SUCCESS] Loki is running on port 3100
)

echo.
echo 🎉 Monitoring Stack Successfully Started!
echo ==========================================
echo.
echo 📊 Access your monitoring services:
echo    • Grafana Dashboard:    http://localhost:3000
echo      - Username: admin
echo      - Password: admin
echo.
echo    • Prometheus:           http://localhost:9090
echo    • Alertmanager:         http://localhost:9093
echo    • Loki:                 http://localhost:3100
echo.
echo 📈 Pre-configured Dashboards:
echo    • Vietnamese ID Card API Monitoring
echo    • System Resource Monitoring
echo    • Application Logs Dashboard
echo.
echo 🔔 Alerts are configured for:
echo    • High API error rates
echo    • Low confidence scores
echo    • High system resource usage
echo    • GPU performance issues
echo.
echo 💡 To stop the monitoring stack:
echo    stop-monitoring.bat
echo.

pause
