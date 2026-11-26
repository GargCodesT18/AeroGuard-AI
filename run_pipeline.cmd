@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo ==================================
echo RUL Prediction Pipeline
echo ==================================
echo Using Python: %PYTHON_BIN%

echo.
echo 1. Downloading official NASA C-MAPSS data...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\download_data.py"
if errorlevel 1 goto :error

echo.
echo 2. Training the multi-source LSTM...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\train.py" --require-cuda --epochs 20 --patience 5 --batch-size 256 --standardize-target --schedule --clip-grad 1 --validation-only
if errorlevel 1 goto :error

echo.
echo 3. Evaluating saved checkpoint...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\evaluate.py" --dataset ALL
if errorlevel 1 goto :error

echo.
echo ==================================
echo Pipeline completed successfully!
echo ==================================
goto :eof

:error
echo.
echo Pipeline failed.
exit /b 1
