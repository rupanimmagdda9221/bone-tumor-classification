@echo off
REM Re-computes test-set metrics and plots, then the with/without-augmentation comparison.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe ( echo Run setup_windows.bat first. & pause & exit /b 1 )
if not exist data\raw\malignant ( echo No dataset found - run download_dataset.bat first. & pause & exit /b 1 )

cd src
..\.venv\Scripts\python evaluate.py
if exist ..\outputs\no_augmentation\best_model.pt ..\.venv\Scripts\python evaluate.py --output-dir ..\outputs\no_augmentation
..\.venv\Scripts\python compare.py
pause
