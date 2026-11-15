@echo off
REM THEIA - one-click start on Windows: opens the website in your browser.
REM First run: creates a private Python environment and installs packages (needs internet once).
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist ".venv\Scripts\python.exe" (
  echo Setting up THEIA for the first time - this takes a few minutes...
  %PY% -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Installation failed. & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
