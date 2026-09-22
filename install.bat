@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "LOG=%~dp0install_log.txt"
echo REMASTRA install %date% %time% > "%LOG%"

rem ---------------------------------------------------------------
rem Langue / Language  (/lang:en ou /lang:fr, transmis par Remastra.exe)
rem ---------------------------------------------------------------
set "L=fr"
set "FROM_LAUNCHER=0"
for %%a in (%*) do (
  if /I "%%~a"=="/lang:en" set "L=en"
  if /I "%%~a"=="/lang:fr" set "L=fr"
  if /I "%%~a"=="/launcher" set "FROM_LAUNCHER=1"
)
if "%L%"=="en" (
  set "T_TITLE=REMASTRA - Installing components"
  set "T_SUB=Installing components - first launch"
  set "T_PYNF=[1/5] Python 3.11 not found: installing it with winget..."
  set "T_PYERR1=ERROR: Python 3.10 to 3.12 is required."
  set "T_PYERR2=Install Python 3.11 from https://www.python.org/downloads/"
  set "T_PYERR3=and tick Add python.exe to PATH, then run Remastra.exe again."
  set "T_PYOK=[1/5] Python found:"
  set "T_VENV=[2/5] Creating the runtime environment..."
  set "T_VENVOK=[2/5] Runtime environment already present"
  set "T_UI=[3/5] Installing the interface and the audio engine..."
  set "T_TORCH=[4/5] Installing PyTorch - large download, please wait..."
  set "T_TORCHERR=  * PyTorch could not be installed: the AI engines will be disabled."
  set "T_AI=[5/5] Installing the AI engines - Demucs v4, DeepFilterNet 3, BS-RoFormer..."
  set "T_AIERR=  * Some AI engines could not be installed - see install_log.txt"
  set "T_MODELS=  Downloading the AI models..."
  set "T_DONE=  Installation complete. REMASTRA is starting."
  set "T_ERR=  ERROR during installation. Details in install_log.txt"
) else (
  set "T_TITLE=REMASTRA - Installation des composants"
  set "T_SUB=Installation des composants - premier lancement"
  set "T_PYNF=[1/5] Python 3.11 introuvable : installation via winget..."
  set "T_PYERR1=ERREUR : Python 3.10 a 3.12 est requis."
  set "T_PYERR2=Installez Python 3.11 depuis https://www.python.org/downloads/"
  set "T_PYERR3=en cochant Add python.exe to PATH, puis relancez Remastra.exe."
  set "T_PYOK=[1/5] Python detecte :"
  set "T_VENV=[2/5] Creation de l'environnement runtime..."
  set "T_VENVOK=[2/5] Environnement runtime existant"
  set "T_UI=[3/5] Installation de l'interface et du moteur audio..."
  set "T_TORCH=[4/5] Installation de PyTorch - volumineux, patience..."
  set "T_TORCHERR=  * PyTorch n'a pas pu etre installe : les moteurs IA seront desactives."
  set "T_AI=[5/5] Installation des moteurs IA - Demucs v4, DeepFilterNet 3, BS-RoFormer..."
  set "T_AIERR=  * Certains moteurs IA n'ont pas pu etre installes - voir install_log.txt"
  set "T_MODELS=  Pre-telechargement des modeles IA..."
  set "T_DONE=  Installation terminee. REMASTRA demarre."
  set "T_ERR=  ERREUR pendant l'installation. Details dans install_log.txt"
)
title !T_TITLE!

echo.
echo   ============================================================
echo      R E M A S T R A   -   AI Audio Remaster Studio
echo      !T_SUB!
echo   ============================================================
echo.

rem ---------------------------------------------------------------
rem 1. Python 3.10 - 3.12
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
  echo !T_PYNF!
  winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  if not defined PY ( py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11" )
)
if not defined PY (
  echo.
  echo  !T_PYERR1!
  echo  !T_PYERR2!
  echo  !T_PYERR3!
  echo ERROR python not found >> "%LOG%"
  pause
  exit /b 1
)
echo !T_PYOK! !PY!
echo python=!PY! >> "%LOG%"

rem ---------------------------------------------------------------
rem 2. Environnement isole "runtime"
rem ---------------------------------------------------------------
if not exist runtime\Scripts\python.exe (
  echo !T_VENV!
  !PY! -m venv runtime >> "%LOG%" 2>&1 || goto :err
) else (
  echo !T_VENVOK!
)
set "VPY=%~dp0runtime\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel --log "%LOG%" -q

rem ---------------------------------------------------------------
rem 3. Interface + moteur DSP
rem ---------------------------------------------------------------
echo !T_UI!
"%VPY%" -m pip install -r requirements.txt --log "%LOG%" || goto :err

rem ---------------------------------------------------------------
rem 4. PyTorch (GPU NVIDIA si present, sinon CPU)
rem ---------------------------------------------------------------
set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>&1 && set "TORCH_IDX=https://download.pytorch.org/whl/cu126"
echo !T_TORCH!
echo   !TORCH_IDX!
"%VPY%" -m pip install "torch==2.7.1" "torchaudio==2.7.1" --index-url !TORCH_IDX! --log "%LOG%"
if errorlevel 1 (
  echo !T_TORCHERR!
  goto :ready
)

rem ---------------------------------------------------------------
rem 5. Moteurs IA
rem ---------------------------------------------------------------
echo !T_AI!
"%VPY%" -m pip install -r requirements-ai.txt --log "%LOG%"
if errorlevel 1 echo !T_AIERR!

echo !T_MODELS!
"%VPY%" -c "from remastra.core import stems; stems._get_model('htdemucs_6s'); stems._get_model('htdemucs'); print('  Demucs OK')" 2>> "%LOG%"
"%VPY%" -c "from remastra.core import denoise; import numpy as np; denoise.deepfilter_denoise(np.zeros((1,48000),'float32'),48000); print('  DeepFilterNet OK')" 2>> "%LOG%"

:ready
echo ok> runtime\.ready
echo.
echo !T_DONE!
echo.
if "%FROM_LAUNCHER%"=="1" ( timeout /t 3 >nul ) else ( start "" "%~dp0Remastra.exe" & pause )
exit /b 0

:err
echo.
echo !T_ERR!
echo ERROR >> "%LOG%"
pause
exit /b 1
