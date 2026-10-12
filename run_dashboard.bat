@echo off
REM Opens the dashboard in your browser (Windows). Double-click to run.
cd /d "%~dp0"
python -m streamlit run epl_value_dashboard.py
pause
