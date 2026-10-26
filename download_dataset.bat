@echo off
REM Downloads the BTXRD dataset (~840 MB, CC BY 4.0) and sorts it into data\raw\benign and data\raw\malignant.
REM Only needed for training or evaluation - the demo works without it.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe ( echo Run setup_windows.bat first. & pause & exit /b 1 )
if not exist data\btxrd mkdir data\btxrd

if not exist data\btxrd\BTXRD\dataset.xlsx (
    echo Downloading BTXRD from figshare ^(about 840 MB^) ...
    curl -L --fail -o data\btxrd\BTXRD.zip https://ndownloader.figshare.com/files/50653575
    if errorlevel 1 ( echo Download failed - check your internet connection. & pause & exit /b 1 )
    echo Extracting ...
    tar -xf data\btxrd\BTXRD.zip -C data\btxrd
    if errorlevel 1 ( echo Extraction failed. & pause & exit /b 1 )
    del data\btxrd\BTXRD.zip
)

.venv\Scripts\python src\prepare_btxrd.py --src data\btxrd\BTXRD
pause
