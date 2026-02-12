@echo off
REM Production startup script for Windows

echo ==========================================
echo CrediTrust Complaint Analyzer
echo Production Startup Script
echo ==========================================

REM Check Python version
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo Warning: .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env with your configuration before continuing.
)

REM Check for vector store
if not exist "vector_store\faiss_index.bin" (
    echo Warning: Vector store not found!
    echo Please run: python src\index_vector_store.py
    exit /b 1
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Start application
echo Starting application...
python app.py
