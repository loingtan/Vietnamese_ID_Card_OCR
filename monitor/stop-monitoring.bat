@echo off
REM Vietnamese ID Card OCR - Monitoring Stack Stop Script (Windows)
REM This script stops the complete monitoring infrastructure on Windows

echo 🛑 Stopping Vietnamese ID Card OCR Monitoring Stack...

REM Stop the monitoring stack
echo [INFO] Stopping monitoring stack...
cd monitoring

REM Stop and remove containers
docker-compose -f docker-compose.monitoring.yml down

echo [SUCCESS] Monitoring stack stopped successfully!

REM Clean up option
if "%1"=="--clean" (
    echo [INFO] Cleaning up monitoring data...
    set /p clean_confirm="Are you sure you want to delete all monitoring data? (y/N): "
    if /i "%clean_confirm%"=="y" (
        rmdir /s /q monitoring\prometheus\data 2>nul
        rmdir /s /q monitoring\grafana\data 2>nul
        rmdir /s /q monitoring\loki\data 2>nul
        rmdir /s /q monitoring\alertmanager\data 2>nul
        mkdir monitoring\prometheus\data
        mkdir monitoring\grafana\data
        mkdir monitoring\loki\data
        mkdir monitoring\alertmanager\data
        echo [SUCCESS] Monitoring data cleaned up
    ) else (
        echo [INFO] Keeping monitoring data
    )
)

echo.
echo ✅ Monitoring Stack Stopped
echo ==========================
echo.
echo The monitoring services have been stopped.
echo.
echo 💡 Available options:
echo    • To restart: start-monitoring.bat
echo    • To clean data: stop-monitoring.bat --clean
echo.

pause
