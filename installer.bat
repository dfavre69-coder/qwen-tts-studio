@echo off
setlocal
chcp 65001 >nul
title Installation de Qwen3-TTS Voice Studio
cd /d "%~dp0"

REM ============================================================================
REM   INSTALLATION DE QWEN3-TTS VOICE STUDIO
REM   Copyright 2026 Denis Favre - SPDX-License-Identifier: Apache-2.0
REM
REM   Cree l'environnement Python "qwen_env" dans ce dossier, installe PyTorch
REM   (GPU NVIDIA, CUDA 12.1) et les dependances, verifie ffmpeg, puis
REM   propose de telecharger le modele (environ 4,5 Go).
REM   Relancable sans risque : il complete ce qui manque.
REM ============================================================================

set "VENV=%~dp0qwen_env"
set "MODELE=Qwen/Qwen3-TTS-12Hz-1.7B-Base"
set "TOKENIZER=Qwen/Qwen3-TTS-Tokenizer-12Hz"
set "INDEX_TORCH=https://download.pytorch.org/whl/cu121"

echo.
echo ======================================================================
echo   INSTALLATION DE QWEN3-TTS VOICE STUDIO
echo ======================================================================
echo.

REM --- 1. Python 3.12 ---------------------------------------------------------
set "PYTHON="
for /f "delims=" %%p in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON=%%p"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYTHON goto :python_ok
echo   [!] Python 3.12 est introuvable sur cet ordinateur.
choice /c ON /m "      L'installer maintenant (winget)"
if errorlevel 2 goto :err_python
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PYTHON%" goto :err_python
:python_ok
echo   [1/5] Python 3.12 : %PYTHON%

REM --- 2. Environnement virtuel ----------------------------------------------
if exist "%VENV%\Scripts\python.exe" goto :venv_ok
echo   [2/5] Creation de l'environnement qwen_env...
"%PYTHON%" -m venv "%VENV%"
if errorlevel 1 goto :err_venv
:venv_ok
set "PY=%VENV%\Scripts\python.exe"
echo   [2/5] Environnement : %VENV%
"%PY%" -m pip install --upgrade pip --quiet

REM --- 3. Paquets ---------------------------------------------------------------
echo.
echo   [3/5] Installation de PyTorch pour GPU NVIDIA (environ 2,5 Go)...
"%PY%" -m pip install -r requirements-torch.txt --index-url %INDEX_TORCH%
if errorlevel 1 goto :err_pip
echo.
echo   [3/5] Installation des autres paquets...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :err_pip

REM --- 4. ffmpeg ----------------------------------------------------------------
echo.
where ffprobe >nul 2>&1
if not errorlevel 1 goto :ffmpeg_ok
echo   [4/5] ffmpeg est absent (necessaire pour le MP3, la video et la fusion).
choice /c ON /m "      L'installer maintenant (winget)"
if errorlevel 2 goto :ffmpeg_suite
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
echo       ffmpeg sera reconnu a la prochaine ouverture d'une fenetre.
goto :ffmpeg_suite
:ffmpeg_ok
echo   [4/5] ffmpeg : present
:ffmpeg_suite

REM --- 5. Modele ----------------------------------------------------------------
echo.
echo   [5/5] Le modele (environ 4,5 Go) peut etre telecharge maintenant ou
echo         automatiquement au premier rendu.
choice /c ON /m "      Le telecharger maintenant"
if errorlevel 2 goto :verification
"%PY%" -c "import sys; from huggingface_hub import snapshot_download as s; [print('OK', s(r)) for r in sys.argv[1:]]" "%MODELE%" "%TOKENIZER%"
if errorlevel 1 echo       [!] Telechargement incomplet : relance installer.bat, il reprendra.

REM --- Verification -------------------------------------------------------------
:verification
echo.
echo ----------------------------------------------------------------------
"%PY%" -c "import torch, qwen_tts; ok=torch.cuda.is_available(); print('   PyTorch', torch.__version__, '| GPU :', torch.cuda.get_device_name(0) if ok else 'AUCUN (le rendu sera tres lent)')"
if errorlevel 1 goto :err_verif
echo ----------------------------------------------------------------------
echo.
echo   Installation terminee. Pour lancer le studio : lancer_studio.bat
echo.
pause
endlocal
exit /b 0


:err_python
echo   [X] Python 3.12 est necessaire : https://www.python.org/downloads/
goto :sortie_erreur
:err_venv
echo   [X] Impossible de creer l'environnement dans %VENV%
goto :sortie_erreur
:err_pip
echo   [X] L'installation d'un paquet a echoue (voir les messages ci-dessus).
echo       Verifie ta connexion Internet, puis relance installer.bat.
goto :sortie_erreur
:err_verif
echo   [X] L'environnement est installe mais torch ou qwen_tts ne se charge pas.
goto :sortie_erreur
:sortie_erreur
echo.
pause
endlocal
exit /b 1
