@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   FireRedVAD Scientific Corpus Validation & Benchmark
echo =======================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

set SCRIPT_DIR=%~dp0
set VALIDATION_SCRIPT=%SCRIPT_DIR%fireredvad_validation\fireredvad_evaluate_corpora.py

if not exist "%VALIDATION_SCRIPT%" (
    echo [ERROR] Script not found: %VALIDATION_SCRIPT%
    pause
    exit /b 1
)

echo Running FireRedVAD Corpus Validation and Benchmark...
echo.
python "%VALIDATION_SCRIPT%" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Validation run failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] FireRedVAD Corpus Validation completed successfully!
pause
