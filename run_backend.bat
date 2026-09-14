@echo off
echo Starting FitPulse FastAPI Backend on port 8003...
cd /d "%~dp0\FitPulse-Health-Anomaly-Detection-from-Fitness Devices"
python -m uvicorn backend:app --host 127.0.0.1 --port 8003 --reload
pause
