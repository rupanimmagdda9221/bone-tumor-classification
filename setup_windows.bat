@echo off
REM One-time setup on Windows: creates .venv and installs PyTorch + project requirements.
REM Uses the NVIDIA GPU build of PyTorch automatically if an NVIDIA driver is found.
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

%PY% --version
if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" during installation, then run this file again.
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment .venv ...
    %PY% -m venv .venv
    if errorlevel 1 ( echo Failed to create .venv & pause & exit /b 1 )
)

.venv\Scripts\python -m pip install --upgrade pip

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo NVIDIA GPU detected - installing CUDA build of PyTorch ...
    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
) else (
    echo No NVIDIA GPU detected - installing CPU build of PyTorch ...
    .venv\Scripts\python -m pip install torch torchvision
)
if errorlevel 1 ( echo PyTorch installation failed & pause & exit /b 1 )

.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Requirements installation failed & pause & exit /b 1 )

echo.
.venv\Scripts\python -c "import torch; print('PyTorch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
echo.
echo Setup complete. Double-click run_demo.bat to start the web demo.
pause
