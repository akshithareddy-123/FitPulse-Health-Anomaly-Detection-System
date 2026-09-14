@echo off
echo ==============================================
echo       Starting FitPulse Analytics Suite
echo ==============================================
start "FitPulse Backend (Port 8003)" cmd /k "call "%~dp0run_backend.bat""
timeout /t 3 /nobreak >nul
start "FitPulse Frontend (Port 8502)" cmd /k "call "%~dp0run_frontend.bat""
echo Both Backend and Frontend services started!
echo Backend:  http://localhost:8003/docs
echo Frontend: http://localhost:8502
echo ==============================================
