#!/usr/bin/env sh
# Opens the dashboard in your browser (macOS / Linux).
cd "$(dirname "$0")" && python3 -m streamlit run epl_value_dashboard.py
