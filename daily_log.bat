@echo off
REM Headless paper-trading run for Windows Task Scheduler. Appends output to daily_log.txt
cd /d "%~dp0"
python epl_value_dashboard.py --log >> daily_log.txt 2>&1
