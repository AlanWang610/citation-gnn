#!/bin/bash

# Simple script to run the Citation Recommendation Web App
echo "Starting Citation Recommendation Web App..."

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. If you want to create one, run:"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
fi

# Run the Flask app
python app.py

echo "Web app has been stopped." 
