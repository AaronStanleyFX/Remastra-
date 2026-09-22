@echo off
setlocal
title REMASTRA - Installation des composants
cd /d "%~dp0"
set LOG=%~dp0install_log.txt
echo REMASTRA install %date% %time% > "%LOG%"

echo.
echo   ============================================================
echo      R E M A S T R A   -   AI Audio Remaster Studio
echo      Installation des composants (premier lancement)
echo   ============================================================
echo.

rem ---------------------------------------------------------------
rem 1. Recherche de Python 3.10 - 3.12
rem ---------------------------------------------------------------
set "PY="
for %%v in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [1/5] Python 3.11 introuvable : installation via winget...
  winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  if not defined PY ( py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11" )
)
if not defined PY (
  echo.
  echo  ERREUR : Python 3.10 a 3.12 est requis.
  echo  Installez Python 3.11 depuis https://www.python.org/downloads/
  echo  en cochant "Add python.exe to PATH", puis relancez Remastra.exe.
  echo ERREUR python introuvable >> "%LOG%"
  pause
  exit /b 1
)
echo [1/5] Python detecte : %PY%
echo python=%PY% >> "%LOG%"

rem ---------------------------------------------------------------
rem 2. Environnement isole "runtime"
rem ---------------------------------------------------------------
if not exist runtime\Scripts\python.exe (
  echo [2/5] Creation de l'environnement runtime...
  %PY% -m venv runtime >> "%LOG%" 2>&1 || goto :err
) else (
  echo [2/5] Environnement runtime existant
)
set "VPY=%~dp0runtime\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel --log "%LOG%" -q

rem ---------------------------------------------------------------
rem 3. Interface + moteur DSP
rem ---------------------------------------------------------------
echo [3/5] Installation de l'interface et du moteur audio...
"%VPY%" -m pip install -r requirements.txt --log "%LOG%" || goto :err

rem ---------------------------------------------------------------
rem 4. PyTorch (GPU NVIDIA si present, sinon CPU)
rem ---------------------------------------------------------------
set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>&1 && set "TORCH_IDX=https://download.pytorch.org/whl/cu126"
echo [4/5] Installation de PyTorch (%TORCH_IDX%)... (volumineux, patience)
"%VPY%" -m pip install "torch==2.7.1" "torchaudio==2.7.1" --index-url %TORCH_IDX% --log "%LOG%"
if errorlevel 1 (
  echo   ! PyTorch n'a pas pu etre installe : les moteurs IA seront desactives.
  goto :ready
)

rem ---------------------------------------------------------------
rem 5. Moteurs IA
rem ---------------------------------------------------------------
echo [5/5] Installation des moteurs IA (Demucs v4, DeepFilterNet 3)...
"%VPY%" -m pip install -r requirements-ai.txt --log "%LOG%"
if errorlevel 1 echo   ! Certains moteurs IA n'ont pas pu etre installes (voir install_log.txt).

echo   Pre-telechargement des modeles IA...
"%VPY%" -c "from remastra.core import stems; stems._get_model('htdemucs_6s'); stems._get_model('htdemucs'); print('  Demucs OK')" 2>> "%LOG%"
"%VPY%" -c "from remastra.core import denoise; import numpy as np; denoise.deepfilter_denoise(np.zeros((1,48000),'float32'),48000); print('  DeepFilterNet OK')" 2>> "%LOG%"

:ready
echo ok> runtime\.ready
echo.
echo   Installation terminee ! REMASTRA va demarrer.
echo.
if /I "%~1"=="/launcher" ( timeout /t 3 >nul ) else ( start "" "%~dp0Remastra.exe" & pause )
exit /b 0

:err
echo.
echo  ERREUR pendant l'installation. Details dans install_log.txt
echo ERREUR >> "%LOG%"
pause
exit /b 1
