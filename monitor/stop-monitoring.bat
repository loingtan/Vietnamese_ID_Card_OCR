@echo off
REM Vietnamese ID Card OCR - Complete Stack Stop Script (Windows)
REM This script stops the complete application and monitoring infrastructure

echo 🛑 Stopping Vietnamese ID Card OCR Complete Stack...

REM Navigate to deployment directory
echo [INFO] Navigating to deployment directory...
cd "..\deployment\docker"

REM Stop the complete stack
echo [INFO] Stopping complete stack...

REM Stop and remove containers
docker-compose down

echo [SUCCESS] Complete stack stopped successfully!

REM Clean up option
if "%1"=="--clean" (
    echo [INFO] Cleaning up data...
    set /p clean_confirm="Are you sure you want to delete all data volumes? (y/N): "
    if /i "%clean_confirm%"=="y" (
        docker-compose down -v
        docker system prune -f
        echo [SUCCESS] All data cleaned up
    ) else (
        echo [INFO] Keeping data volumes
    )
)

echo.
echo ✅ Complete Stack Stopped Successfully!
echo.
echo 💡 To restart the stack:
echo    start-monitoring.bat
echo.
echo 🧹 To stop and clean all data:
echo    stop-monitoring.bat --clean
echo.

pause

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
