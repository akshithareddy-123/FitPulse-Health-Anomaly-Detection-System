@echo off
echo Starting FitPulse Streamlit Frontend on port 8502...
cd /d "%~dp0\FitPulse-Health-Anomaly-Detection-from-Fitness Devices"
python -m streamlit run app.py --server.port 8502
pause
