@echo off
REM Double-click launcher for the BESS P50/P75/P90 app.
cd /d "%~dp0"
echo Starting BESS P50 / P75 / P90 app ...
echo The browser will open at http://localhost:8501
echo Keep this window open while using the app. Close it to stop.
python -m streamlit run bess_app_v4_fixed.py --server.port 8501 --browser.gatherUsageStats false
pause
