@echo off
ECHO Starting Citation Recommendation Web App...

:: Check if virtual environment exists and activate it
IF EXIST "venv" (
    ECHO Activating virtual environment...
    CALL venv\Scripts\activate.bat
) ELSE (
    ECHO Virtual environment not found. If you want to create one, run:
    ECHO python -m venv venv
    ECHO venv\Scripts\activate.bat
    ECHO pip install -r requirements.txt
)

:: Run the Flask app
python app.py

ECHO Web app has been stopped.
PAUSE 
