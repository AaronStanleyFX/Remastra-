@echo off
rem Construit une version autonome (dossier dist\REMASTRA\REMASTRA.exe, sans Python requis).
rem Prerequis : avoir lance Remastra.exe une premiere fois (environnement runtime installe).
cd /d "%~dp0"
set "VPY=%~dp0runtime\Scripts\python.exe"
"%VPY%" -m pip install pyinstaller
"%VPY%" -m PyInstaller --noconfirm --clean --windowed --name REMASTRA ^
  --icon remastra\assets\icon.ico ^
  --add-data "remastra\assets;remastra\assets" ^
  --collect-all demucs --collect-all df --collect-all pedalboard --collect-all soxr ^
  --collect-submodules remastra ^
  run_remastra.py
echo.
echo Version autonome : dist\REMASTRA\REMASTRA.exe
pause
