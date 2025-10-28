# SLM for Academic Calendar Interpretation

This project is a lightweight pipeline that ingests academic calendar PDFs and images, extracts structured events, and answers natural language queries about them. The entire process runs locally without relying on external LLM APIs.

## Features

- **Data Ingestion:** Supports both PDF and image files.
- **Text Extraction:** Uses PyMuPDF for PDFs and Tesseract OCR for images.
- **Structured Data:** Parses raw text into a structured CSV format.
- **Date Normalization:** Normalizes various date formats into ISO 8601.
- **Event Categorization:** Assigns a category to each event (e.g., `exam`, `holiday`).
- **QA System:** A simple TF-IDF-based retrieval model to answer natural language queries.
- **Web Interface:** A Flask-based UI to upload files and ask questions.

## Project Structure

```
├── app/
│   ├── app.py              # Flask web application
│   └── templates/
│       └── index.html      # HTML template for the UI
├── data/
│   ├── calendar.png        # Sample calendar image
│   └── ...                 # Place your calendar files here
├── src/
│   ├── ingest_data.py      # Ingests data from PDFs and images
│   ├── normalize_dates.py  # Normalizes dates
│   ├── categorize_events.py # Categorizes events
│   └── baseline_retrieval.py # Builds the TF-IDF model
├── notebooks/
│   └── (Exploratory notebooks)
├── run_pipeline.py         # Master script to run the data processing pipeline
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup and Usage

### 1. Prerequisites

- **Python 3.8+**
- **Tesseract OCR:** You must have Tesseract installed on your system.
  - **macOS:** `brew install tesseract`
  - **Ubuntu:** `sudo apt-get install tesseract-ocr`
  - **Windows:** Download from the [official Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki).

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### 3. Add Your Calendar Files

Place your academic calendar files (PDFs or images) into the `data/` directory.

### 4. Run the Data Processing Pipeline

To process the files in the `data/` directory and build the search model, run the master pipeline script:

```bash
python3 run_pipeline.py
```

This will create a `data/calendar_events.csv` file and the TF-IDF model files (`data/tfidf_vectorizer.joblib` and `data/tfidf_matrix.joblib`).

### 5. Run the Web Application

Start the Flask web server:

```bash
python3 app/app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.
