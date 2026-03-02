@echo off
title Crowd Anomaly Detection - Demo Launcher
color 0A

echo ============================================================
echo   Abnormal Crowd Behaviour Detection System
echo   Demo Launcher
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11 or higher from https://python.org
    pause
    exit /b 1
)

REM Check dependencies are installed
echo Checking dependencies...
python -c "import streamlit, cv2, joblib, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Dependencies not found. Installing now (this only happens once)...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies. Please contact support.
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully.
    echo.
)

REM Check model artifact exists
if not exist "artifacts\models\shanghaitech_windowed_rf.joblib" (
    echo ERROR: Model file not found.
    echo Expected location: artifacts\models\shanghaitech_windowed_rf.joblib
    echo Please ensure the full project folder was copied correctly.
    pause
    exit /b 1
)

echo All checks passed.
echo.
echo Starting the demo application...
echo A browser window will open automatically in a few seconds.
echo.
echo To stop the demo: close this window or press Ctrl+C
echo ============================================================
echo.

python -m streamlit run scripts/crowd_anomaly_demo.py --server.headless false --browser.gatherUsageStats false

pause
