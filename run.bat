@echo off
echo ================================================
echo  ModelRouter — Team Dynamic IP
echo  Intelligent LLM Query Routing System
echo ================================================
echo.

set PYTHON=C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe

echo [1/2] Installing dependencies...
%PYTHON% -m pip install -r backend/requirements.txt > nul 2>&1

echo [2/2] Starting ModelRouter engine...
echo.
echo  API:      http://localhost:8000
echo  Dashboard: http://localhost:8000/dashboard
echo  Docs:     http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop
echo.

%PYTHON% backend/main.py

pause
