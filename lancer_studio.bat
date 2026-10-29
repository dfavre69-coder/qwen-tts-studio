@echo off
REM Copyright 2026 Denis Favre - SPDX-License-Identifier: Apache-2.0
REM Lance Qwen3-TTS Voice Studio avec son environnement Python.
cd /d "%~dp0"
if not exist "%~dp0qwen_env\Scripts\python.exe" (
    echo.
    echo   L'environnement n'est pas installe : lance d'abord installer.bat
    echo.
    pause
    exit /b 1
)
"%~dp0qwen_env\Scripts\python.exe" "%~dp0qwen_tts_studio.py"
if errorlevel 1 pause
