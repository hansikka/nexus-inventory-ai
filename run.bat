@echo off
setlocal
cd /d "%~dp0"
title Nexus Inventory AI Dashboard

cls
echo =======================================================
echo    NEXUS INVENTORY AI - Demand Forecast Dashboard
echo =======================================================
echo.
echo Starting application...
echo Opening http://localhost:5005 in your browser...
echo.

start "" cmd /c "ping 127.0.0.1 -n 3 >nul && start http://localhost:5005"

python backend\app.py
if %errorlevel% neq 0 (
    py -3 backend\app.py
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Could not start the server.
    echo Make sure Python is installed and in your PATH.
    echo.
    pause
)