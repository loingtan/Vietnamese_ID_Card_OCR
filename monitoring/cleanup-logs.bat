@echo off
REM Vietnamese ID Card OCR - Log Cleanup and Rotation Script (Windows)
REM This script manages log rotation and cleanup for the monitoring system

echo Vietnamese ID Card OCR - Log Cleanup and Rotation
echo ================================================

REM Configuration
set LOG_DIR=logs
set LOG_RETENTION_DAYS=30
set MAX_LOG_SIZE_MB=100

REM Create necessary directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "backups" mkdir "backups"

echo [INFO] Starting log cleanup and rotation...

REM Rotate application logs
echo [INFO] Checking application logs for rotation...

if exist "%LOG_DIR%\api.log" (
    for %%F in ("%LOG_DIR%\api.log") do (
        set size=%%~zF
        set /a size_mb=!size!/1024/1024
        if !size_mb! gtr %MAX_LOG_SIZE_MB% (
            set timestamp=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
            set timestamp=!timestamp: =0!
            move "%LOG_DIR%\api.log" "%LOG_DIR%\api.log.!timestamp!"
            echo. > "%LOG_DIR%\api.log"
            echo [INFO] Rotated api.log
        )
    )
)

if exist "%LOG_DIR%\error.log" (
    for %%F in ("%LOG_DIR%\error.log") do (
        set size=%%~zF
        set /a size_mb=!size!/1024/1024
        if !size_mb! gtr %MAX_LOG_SIZE_MB% (
            set timestamp=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
            set timestamp=!timestamp: =0!
            move "%LOG_DIR%\error.log" "%LOG_DIR%\error.log.!timestamp!"
            echo. > "%LOG_DIR%\error.log"
            echo [INFO] Rotated error.log
        )
    )
)

if exist "%LOG_DIR%\model.log" (
    for %%F in ("%LOG_DIR%\model.log") do (
        set size=%%~zF
        set /a size_mb=!size!/1024/1024
        if !size_mb! gtr %MAX_LOG_SIZE_MB% (
            set timestamp=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
            set timestamp=!timestamp: =0!
            move "%LOG_DIR%\model.log" "%LOG_DIR%\model.log.!timestamp!"
            echo. > "%LOG_DIR%\model.log"
            echo [INFO] Rotated model.log
        )
    )
)

if exist "%LOG_DIR%\metrics.log" (
    for %%F in ("%LOG_DIR%\metrics.log") do (
        set size=%%~zF
        set /a size_mb=!size!/1024/1024
        if !size_mb! gtr %MAX_LOG_SIZE_MB% (
            set timestamp=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
            set timestamp=!timestamp: =0!
            move "%LOG_DIR%\metrics.log" "%LOG_DIR%\metrics.log.!timestamp!"
            echo. > "%LOG_DIR%\metrics.log"
            echo [INFO] Rotated metrics.log
        )
    )
)

REM Clean old log files (simplified - remove files older than 30 days)
echo [INFO] Cleaning old log files...
forfiles /p "%LOG_DIR%" /s /m *.log.* /d -%LOG_RETENTION_DAYS% /c "cmd /c del @path" 2>nul

REM Create backup if requested
if "%1"=="--backup" (
    echo [INFO] Creating backup of monitoring configuration...
    set backup_dir=backups\monitoring_backup_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    set backup_dir=!backup_dir: =0!
    mkdir "!backup_dir!" 2>nul
    
    if exist "monitoring\prometheus\prometheus.yml" copy "monitoring\prometheus\prometheus.yml" "!backup_dir!\" >nul
    if exist "monitoring\prometheus\alert-rules.yml" copy "monitoring\prometheus\alert-rules.yml" "!backup_dir!\" >nul
    if exist "monitoring\alertmanager\alertmanager.yml" copy "monitoring\alertmanager\alertmanager.yml" "!backup_dir!\" >nul
    if exist "monitoring\loki\loki-config.yml" copy "monitoring\loki\loki-config.yml" "!backup_dir!\" >nul
    
    echo [SUCCESS] Backup created in !backup_dir!
)

REM Clean old backups
echo [INFO] Cleaning old backups...
forfiles /p "backups" /s /m *.* /d -7 /c "cmd /c del @path" 2>nul

REM Generate report
echo [INFO] Generating cleanup report...
set report_file=%LOG_DIR%\cleanup_report_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set report_file=%report_file: =0%

(
    echo Vietnamese ID Card OCR - Cleanup Report
    echo =======================================
    echo Date: %date% %time%
    echo.
    echo Log Directory Status:
    dir "%LOG_DIR%" 2>nul
    echo.
    echo Monitoring Data Status:
    if exist "monitoring\prometheus\data" echo Prometheus: && dir "monitoring\prometheus\data" /s /-c 2>nul | find "bytes"
    if exist "monitoring\grafana\data" echo Grafana: && dir "monitoring\grafana\data" /s /-c 2>nul | find "bytes"
    if exist "monitoring\loki\data" echo Loki: && dir "monitoring\loki\data" /s /-c 2>nul | find "bytes"
    echo.
    echo Cleanup Settings:
    echo   Log retention: %LOG_RETENTION_DAYS% days
    echo   Max log size: %MAX_LOG_SIZE_MB% MB
) > "%report_file%"

echo [SUCCESS] Cleanup completed successfully!
echo [INFO] Cleanup report saved to %report_file%
echo.
echo [INFO] To schedule automatic cleanup, add this script to Windows Task Scheduler
echo [INFO] Recommended: Run daily at 2 AM with --backup option
echo.

pause
