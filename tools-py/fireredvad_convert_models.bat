@echo off
setlocal
echo =======================================================
echo   FireRedVAD ESP32-P4 - Model Converter (ESPM v3)
echo =======================================================
echo.
echo Converting PyTorch models to ESPM v3 container format...
echo Output: models/espm/
echo.

cd /d "%~dp0"

echo [1/3] Generating FP32 models...
python fireredvad_models_converter\fireredvad_convert_models.py --all --output-dir ..\models\espm
if errorlevel 1 goto :error

echo.
echo [2/3] Generating INT8 per-tensor models...
python fireredvad_models_converter\fireredvad_convert_models.py --all --output-dir ..\models\espm --quantize-int8
if errorlevel 1 goto :error

echo.
echo [3/3] Generating INT8 per-channel models...
python fireredvad_models_converter\fireredvad_convert_models.py --all --output-dir ..\models\espm --quantize-int8-per-ch
if errorlevel 1 goto :error

echo.
echo =======================================================
echo Done! All models successfully exported to models/espm/
echo =======================================================
pause
exit /b 0

:error
echo.
echo [ERROR] FireRedVAD model conversion failed.
pause
exit /b 1
