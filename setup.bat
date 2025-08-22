@echo off
echo Starting MCP News Analysis Project Setup...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Run the setup script
python setup.py

echo.
echo Setup script completed!
echo.
pause
