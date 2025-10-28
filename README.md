# SLM for Academic Calendar Interpretation

This project is a lightweight, from-scratch pipeline for interpreting academic calendars from PDFs or scanned images. It can answer natural language queries about academic schedules (e.g., "When is the last date to drop a course?") without relying on large, pre-trained language models or external APIs.

## Features

-   **OCR Data Ingestion:** Extracts text from calendar images using Tesseract.
-   **Structured Data Conversion:** Parses raw text into a structured CSV format.
-   **Date Normalization:** Converts various date formats into standardized start and end dates.
-   **Baseline Search Model:** Uses a TF-IDF model to find the most relevant calendar events for a given query.
-   **Web Interface:** A simple Flask application provides a user-friendly interface for searching the calendar.

## Prerequisites

Before you begin, you need to have **Tesseract OCR** installed on your system.

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

**On macOS (using Homebrew):**
```bash
brew install tesseract
```

**On Windows:**
Download and install Tesseract from the [official repository](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add the Tesseract installation directory to your system's `PATH`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChinmaySinha/SLM-for-Academic-Calendar-Interpretation.git
    cd SLM-for-Academic-Calendar-Interpretation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

The application consists of two main parts: the data processing pipeline and the web server.

### 1. Run the Data Processing Pipeline

Before you can run the web application, you need to process the source data and build the search model.

```bash
# This script will run all the necessary data processing steps in order.
python3 src/parse_ocr.py
python3 src/normalize_dates.py
python3 src/categorize_events.py
python3 src/baseline_retrieval.py
```

This will generate the following files in the `data/` directory:
-   `calendar.csv`
-   `calendar_normalized.csv`
-   `calendar_processed.csv`
-   `vectorizer.joblib` (the TF-IDF vectorizer)
-   `tfidf_matrix.joblib` (the TF-IDF matrix)

### 2. Start the Web Server

Once the data processing is complete, you can start the Flask web server:

```bash
python3 app/app.py
```

The application will be available at `http://127.0.0.1:5001`.

## Expanding the Dataset

You can improve the accuracy and coverage of the model by adding more academic calendars to the dataset.

1.  **Add New Calendar Files:**
    -   Place your new calendar images (e.g., `.png`, `.jpg`) or PDFs in the `data/` directory.

2.  **Update the Ingestion Script:**
    -   For now, the data ingestion is handled manually by running Tesseract on a single file. To add a new file, you would first run Tesseract on it:
        ```bash
        # For an image file
        tesseract data/new_calendar.png data/ocr_output.txt

        # For a PDF, you would first need to convert it to images, or use a PDF-to-text tool.
        ```
    -   *Note: In the future, this process can be automated to handle multiple files and formats.*

3.  **Re-run the Data Processing Pipeline:**
    -   After updating `data/ocr_output.txt` with the text from your new calendar, re-run the entire pipeline as described in the "Running the Application" section to rebuild the model with the new data.

This will update the search index and make the new calendar events available in the web application.
