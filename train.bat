@echo off
REM Trains the ResNet50 model (results go to outputs\). Extra options are passed through, e.g.:
REM   train.bat --batch-size 16
REM   train.bat --no-augment --output-dir ..\outputs\no_augmentation
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe ( echo Run setup_windows.bat first. & pause & exit /b 1 )
if not exist data\raw\malignant ( echo No dataset found - run download_dataset.bat first. & pause & exit /b 1 )

cd src
..\.venv\Scripts\python train.py %*
pause
