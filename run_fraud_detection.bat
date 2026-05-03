@echo off
:: =============================================================================
::  FRAUD DETECTION SYSTEM — ONE-CLICK LAUNCHER
::  Generated for: files/ folder
::
::  Usage:
::    1. Place this .bat file INSIDE the "files" folder (same level as run.py)
::    2. Double-click it   — OR —   right-click > "Run as administrator"
::
::  What this script does:
::    Step 1  Verifies Python is installed
::    Step 2  Navigates to the correct folder
::    Step 3  Installs all required Python packages via pip
::    Step 4  Trains the fraud detection model (skipped if already trained)
::    Step 5  Starts the FastAPI server on http://localhost:8000
::    Step 6  Opens the Swagger UI in your browser
:: =============================================================================

title Fraud Detection System — Launcher
color 0A

echo.
echo ==============================================================
echo   FRAUD DETECTION API — ONE-CLICK LAUNCHER
echo ==============================================================
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: Change directory to the folder where this .bat file lives
:: (so all relative paths in run.py resolve correctly)
:: ─────────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
echo [INFO] Working directory: %CD%
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 1 — Verify Python is available
:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 1] Checking for Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Python was not found on this machine.
    echo.
    echo  Please install Python 3.9 or later from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 2 — Verify key project files are present
:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 2] Verifying project files...

if not exist "run.py" (
    echo  [ERROR] run.py not found in %CD%
    echo  Make sure this .bat file is in the same folder as run.py
    pause
    exit /b 1
)
if not exist "fraud_detection_system.py" (
    echo  [WARNING] fraud_detection_system.py not found — training will fail if model is absent.
)
if not exist "api_server.py" (
    echo  [ERROR] api_server.py not found. Cannot start the API server.
    pause
    exit /b 1
)
echo   All required files found.
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 3 — Upgrade pip silently
:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 3] Upgrading pip (silent)...
python -m pip install --upgrade pip --quiet
echo   pip is up to date.
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 4 — Install required packages
:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 4] Installing / verifying required packages...
echo   (This may take a minute on first run)
echo.

python -m pip install --quiet --upgrade ^
    fastapi ^
    "uvicorn[standard]" ^
    pydantic ^
    joblib ^
    scikit-learn ^
    numpy ^
    pandas ^
    imbalanced-learn ^
    matplotlib ^
    seaborn

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [WARNING] Some packages may have failed to install.
    echo  Continuing anyway — the server will report missing imports on startup.
    echo.
) else (
    echo   All packages installed successfully.
)
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 5 — Run the one-click launcher (trains model if needed + starts server)
:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 5] Launching Fraud Detection API via run.py...
echo.
echo  -------------------------------------------------------
echo   Server will be available at:
echo     http://localhost:8000/docs    ^<-- Swagger UI (test here)
echo     http://localhost:8000/health  ^<-- Health check
echo     http://localhost:8000/predict ^<-- POST prediction endpoint
echo   Dashboard: open dashboard.html in your browser
echo  -------------------------------------------------------
echo.
echo  Press CTRL+C to stop the server.
echo.

:: Small pause so the user can read the banner before uvicorn logs start
timeout /t 2 /nobreak >nul

:: Open browser to Swagger UI (non-blocking)
start "" "http://localhost:8000/docs"

:: Open the local dashboard.html if it exists (non-blocking)
if exist "dashboard.html" (
    start "" "%CD%\dashboard.html"
)

:: Start the server — this blocks until CTRL+C
python run.py

:: ─────────────────────────────────────────────────────────────────────────────
:: SERVER STOPPED
:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo ==============================================================
echo   Server stopped. Press any key to close this window.
echo ==============================================================
pause >nul
