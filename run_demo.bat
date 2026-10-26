@echo off
REM Launch the online-ready Flask website locally. Keep this window open while the app runs.
setlocal
cd /d "%~dp0"
echo [Bone Tumor Classification] Project folder: %CD%
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Python environment was not found in this project folder.
  echo Please double-click setup_windows.bat in THIS SAME folder and wait for Setup complete.
  pause
  exit /b 1
)
if not exist "outputs\best_model.pt" (
  echo ERROR: outputs\best_model.pt is missing. Re-extract the complete ZIP.
  pause
  exit /b 1
)
echo [Bone Tumor Classification] Starting website on http://127.0.0.1:7860
.venv\Scripts\python.exe web_app.py
if errorlevel 1 (
  echo.
  echo The website stopped because of an error shown above. Take a screenshot of this window.
)
pause
