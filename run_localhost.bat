@echo off
setlocal enabledelayedexpansion
title QuantAI AI - Localhost Launcher

:: Ensure we run from the project root directory
cd /d "%~dp0"

echo =====================================================================
echo                     QuantAI AI - Localhost Launcher
echo =====================================================================
echo.

:: 1. Detect Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=py -3.10"
    ) else (
        if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
            set "PY_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
        ) else (
            echo [ERROR] Python not found in PATH or standard installation directory.
            echo Please install Python 3.10 and add it to your system PATH.
            pause
            exit /b 1
        )
    )
)

echo [OK] Using Python: %PY_CMD%
%PY_CMD% --version

:: 2. Ensure frontend dist exists, build if missing
if not exist "frontend\dist\index.html" (
    echo.
    echo [INFO] Frontend build not found. Building production frontend...
    where npm >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        cd frontend
        call npm run build
        cd ..
    ) else (
        echo [WARNING] npm not found. Make sure Node.js is installed if frontend build is needed.
    )
)

echo.
echo =====================================================================
echo Select launch mode:
echo   [1] Unified Mode (FastAPI + Built Frontend on http://localhost:8000) [Default]
echo   [2] Dev Mode (FastAPI on :8000 + Vite Dev Server on http://localhost:5173)
echo   [3] Run Automated Test Suite (55 tests)
echo =====================================================================
echo.
choice /c 123 /n /t 5 /d 1 /m "Enter choice [1, 2, or 3] (default in 5s: 1): "
set "MODE=!ERRORLEVEL!"

if "!MODE!"=="3" (
    echo.
    echo [INFO] Running test suite...
    %PY_CMD% -m pytest tests/
    echo.
    pause
    exit /b 0
)

if "!MODE!"=="2" (
    echo.
    echo [INFO] Starting Backend on http://localhost:8000...
    start "QuantAI Backend (Port 8000)" %PY_CMD% -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
    
    echo [INFO] Starting Vite Frontend on http://localhost:5173...
    cd frontend
    start "QuantAI Frontend (Port 5173)" cmd /k "npm run dev"
    cd ..
    
    echo [INFO] Waiting 3 seconds before opening browser...
    timeout /t 3 /nobreak >nul
    start http://localhost:5173
    
    echo.
    echo =====================================================================
    echo QuantAI AI Dev Environment is running!
    echo   - Frontend:    http://localhost:5173
    echo   - Backend API: http://localhost:8000
    echo   - API Docs:    http://localhost:8000/docs
    echo Close the opened console windows to stop the servers.
    echo =====================================================================
    exit /b 0
)

:: Default: Mode 1 (Unified Mode)
echo.
echo [INFO] Launching QuantAI AI Unified Server on http://localhost:8000...
echo [INFO] Serving API, Database Analytics, ML Engine, and Frontend at http://localhost:8000
echo.

:: Open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Start Uvicorn in current window
%PY_CMD% -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
