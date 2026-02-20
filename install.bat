@echo off
REM ACE-Step Captioner Installation Script
REM This script sets up a uv virtual environment and installs all dependencies

echo ========================================
echo ACE-Step Captioner Installation
echo ========================================
echo.

REM Check if uv is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv is not installed or not in PATH.
    echo.
    echo Please install uv first:
    echo   Option 1: Install via pip:   pip install uv
    echo   Option 2: Install standalone: https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

echo [INFO] uv found: OK
echo.

REM Check if pyproject.toml exists
if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found in current directory.
    echo Please run this script from the project root directory.
    echo.
    pause
    exit /b 1
)

echo [INFO] Creating virtual environment with uv...
echo [INFO] This may take several minutes depending on your connection.
echo.

REM Create virtual environment and sync dependencies
uv sync

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to create virtual environment or install dependencies.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Installation Complete!
echo ========================================
echo.
echo The virtual environment has been created at: .venv
echo.
echo To use the captioner:
echo   1. Run: run_captioner.bat
echo   2. Or activate the environment manually:
echo      .venv\Scripts\activate
echo.
pause
