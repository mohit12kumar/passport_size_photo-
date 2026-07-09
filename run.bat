@echo off
title AI Passport Studio Launcher
cls

echo ===================================================
echo             📷 AI PASSPORT STUDIO 📷
echo ===================================================
echo.

:: 1. Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo [OK] Found virtual environment. Activating...
    call venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment venv not found. Using system Python.
)
echo.

:: 2. Present options to the user
echo Please select which application to launch:
echo [1] Standard Web Studio (FastAPI Backend + HTML/JS Frontend) [Recommended]
echo [2] Streamlit Web Studio (Alternative UI)
echo [3] Run Both (FastAPI server in background + Streamlit UI)
echo [4] Exit
echo.

set /p choice="Enter option (1-4) [default is 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" goto run_fastapi
if "%choice%"=="2" goto run_streamlit
if "%choice%"=="3" goto run_both
if "%choice%"=="4" goto end
goto invalid_choice

:run_fastapi
echo.
echo Starting FastAPI Web Server (http://127.0.0.1:8000)...
echo Opening browser...
start http://127.0.0.1:8000/
python app.py
goto end

:run_streamlit
echo.
echo Starting Streamlit Web App...
streamlit run streamlit_app.py
goto end

:run_both
echo.
echo Starting FastAPI Web Server in a new window...
if exist venv\Scripts\activate.bat (
    start "FastAPI Server" cmd /c "call venv\Scripts\activate.bat && python app.py"
) else (
    start "FastAPI Server" cmd /c "python app.py"
)
:: Give FastAPI server 2 seconds to start up
timeout /t 2 >nul
echo Starting Streamlit Web App...
streamlit run streamlit_app.py
goto end

:invalid_choice
echo [Error] Invalid choice selected. Exiting.
pause
goto end

:end
