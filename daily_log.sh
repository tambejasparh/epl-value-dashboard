#!/usr/bin/env sh
# Headless paper-trading run for cron. Appends output to daily_log.txt
cd "$(dirname "$0")" && python3 epl_value_dashboard.py --log >> daily_log.txt 2>&1
