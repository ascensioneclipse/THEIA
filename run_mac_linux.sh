#!/usr/bin/env bash
# THEIA - one-click start on macOS / Linux: opens the website in your browser.
# First run: creates a private Python environment and installs packages (needs internet once).
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ ! -x .venv/bin/python ]; then
  echo "Setting up THEIA for the first time - this takes a few minutes..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m streamlit run streamlit_app.py "$@"
