@echo off
setlocal enabledelayedexpansion

rem ── One-time setup (venv + deps) ─────────────────────────────────────────

echo ==================================================
echo ACE-Step Toolkit
echo ==================================================
echo.

echo Checking for uv...
uv --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Installing uv...
    pip install uv
    if !errorlevel! neq 0 (
        echo [ERROR] Could not install uv. Ensure python and pip are in PATH.
        pause
        exit /b 1
    )
)

if not exist ".venv" (
    echo Creating virtual environment with Python 3.11...
    uv venv .venv --python 3.11
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Syncing dependencies...
uv sync
if !errorlevel! neq 0 (
    echo [ERROR] Failed to sync dependencies.
    pause
    exit /b 1
)

echo.
echo [Ready]
echo.

rem ── Folder selection ──────────────────────────────────────────────────────

:FOLDER_SELECTION
echo.
echo ==================================================
echo ACE-Step Toolkit - Project Folder Selection
echo ==================================================
echo.
echo This folder should contain:
echo   - metadata.csv (optional, from Mixxx export)
echo   - audio/       (subfolder with audio files)
echo.
set /p BASE_FOLDER="Enter project folder path: "

rem Remove quotes if user dragged and dropped
set "BASE_FOLDER=!BASE_FOLDER:"=!"

rem Check if folder exists
if not exist "!BASE_FOLDER!" (
    echo [ERROR] Folder not found: !BASE_FOLDER!
    goto FOLDER_SELECTION
)

rem Check if audio subfolder exists
if not exist "!BASE_FOLDER!\audio" (
    echo [ERROR] No 'audio' subfolder found in: !BASE_FOLDER!
    goto FOLDER_SELECTION
)

echo [OK] Project folder: !BASE_FOLDER!
echo.

rem ── Main menu loop ────────────────────────────────────────────────────────

:MENU
echo.
echo ==================================================
echo ACE-Step Toolkit
echo ==================================================
echo.
echo Project: !BASE_FOLDER!
echo.
echo   1. Download Lyrics
echo   2. Caption Audio Files
echo   3. Build Dataset JSON  (merges captions + Mixxx CSV)
echo   4. Change Project Folder
echo   5. Exit
echo.
set /p CHOICE="Enter choice (1-5): "

if "!CHOICE!"=="1" (
    set SCRIPT=scripts\download_lyrics.py
    set TOOL=Lyrics Downloader
    set CUDA=0
) else if "!CHOICE!"=="2" (
    set SCRIPT=scripts\caption_audio.py
    set TOOL=ACE-Step Captioner
    set CUDA=1
) else if "!CHOICE!"=="3" (
    set SCRIPT=scripts\build_dataset.py
    set TOOL=Build Dataset JSON
    set CUDA=0
) else if "!CHOICE!"=="4" (
    goto FOLDER_SELECTION
) else if "!CHOICE!"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo [ERROR] Invalid choice.
    goto MENU
)

echo.
echo ==================================================
echo !TOOL!
echo ==================================================
echo.

if "!CUDA!"=="1" (
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No CUDA\"}')" 2>nul
    echo.
)

python !SCRIPT! "!BASE_FOLDER!"
set SCRIPT_EXIT=!errorlevel!

echo.
echo ==================================================
if !SCRIPT_EXIT! equ 0 (
    echo !TOOL! completed successfully.
) else (
    echo !TOOL! exited with error code !SCRIPT_EXIT!.
)
echo ==================================================
echo.
echo Press any key to return to menu...
pause >nul

goto MENU
