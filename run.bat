@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)
python mic_denoiser.py
if errorlevel 1 (
    echo.
    echo [!] Failed to start. Try: pip install -r requirements.txt
    pause
)
