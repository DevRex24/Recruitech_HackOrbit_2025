# Resume Optimizer

AI-Powered Resume Analysis and Optimization Web App

## Overview

Resume Optimizer is a web application that leverages machine learning and natural language processing to analyze your resume against a job description. It provides actionable suggestions to improve your resume's fit for specific roles and enhances its compatibility with Applicant Tracking Systems (ATS).

## Features

- Upload your resume (PDF, DOCX, or TXT) or paste text directly
- Input or select a job description (with built-in templates for common roles)
- Get an overall fit score based on:
  - Keyword matching
  - Semantic similarity
  - Skills alignment
  - Machine learning predictions
- Receive detailed feedback and suggestions for improvement
- ATS compatibility analysis and formatting tips
- Career field recommendations based on your resume
- User-friendly web interface built with Flask

## Installation

### Requirements
- Python 3.11+

### Dependencies
All required dependencies are listed in `pyproject.toml`. Key packages include:
- Flask
- Flask-SQLAlchemy
- Gunicorn
- NLTK
- NumPy
- Pandas
- Psycopg2-binary
- PyPDF2
- python-docx
- scikit-learn
- Werkzeug

Install dependencies with:
```bash
pip install -r requirements.txt
```
Or, if using Poetry:
```bash
poetry install
```

## Usage

1. Clone the repository and navigate to the `ResumeOptimizer` directory:
   ```bash
   git clone <repo-url>
   cd ResumeOptimizer
   ```
2. Start the Flask app:
   ```bash
   python app.py
   ```
   The app will run on [http://localhost:5000](http://localhost:5000).
3. Open your browser and go to [http://localhost:5000](http://localhost:5000).
4. Upload your resume and job description, then click "Analyze Resume Fit" to get your results.

## Project Structure

- `app.py` - Main Flask application
- `ml_analyzer.py` - Machine learning and NLP analysis logic
- `generate_resume_dataset.py` - Script for generating or processing resume datasets
- `templates/` - HTML templates (main UI in `index.html`)
- `static/` - Static assets (CSS, JS)
- `uploads/` - Temporary storage for uploaded files
- Model files (`*.pkl`) - Pre-trained ML models
