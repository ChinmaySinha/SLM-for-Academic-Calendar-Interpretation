# SLM for Academic Calendar Interpretation

This project is a lightweight, from-scratch pipeline for interpreting academic calendars from PDFs or images. The system uses a dual-stage Small Language Model (SLM) pipeline to answer natural language queries locally without relying on external APIs.

## Features

- **Data Ingestion:** Supports both PDF and image files.
- **Text Extraction:** Uses PyMuPDF for PDFs and Tesseract OCR for images.
- **Structured Data:** Parses raw text into a structured CSV format.
- **Date Normalization:** Normalizes various date formats into ISO 8601.
- **SLM Pipeline:** A semantic search system using `all-MiniLM-L6-v2` embeddings with a FAISS index for retrieval, and `Flan-T5-Small` for generating natural language answers.
- **Web Interface:** A Flask-based UI with a dark theme to ask questions.

## Project Structure

```
├── app/
│   ├── app.py              # Flask web application
│   └── templates/
│       └── index.html      # HTML template for the UI
├── data/
│   └── ...                 # Place your calendar files here
├── scripts/
│   ├── fine_tune_embedder.py # (Placeholder) for fine-tuning the embedder
│   └── fine_tune_flan_lora.py  # (Placeholder) for fine-tuning Flan-T5
├── src/
│   ├── ingest_data.py      # Ingests data from PDFs and images
│   ├── normalize_dates.py  # Normalizes dates
│   ├── categorize_events.py # Categorizes events
│   └── baseline_retrieval.py # Builds the FAISS index
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

First, run the initial data processing steps to generate `data/calendar_events.csv`:
```bash
python3 run_pipeline.py
```

Next, build the FAISS search index:
```bash
python3 src/baseline_retrieval.py
```
This will download the `all-MiniLM-L6-v2` model and create `data/faiss.index` and `data/embeddings.npy`.

### 5. Run the Web Application

Start the Flask web server:
```bash
python3 app/app.py
```
**Note:** The `Flan-T5-Small` model will be downloaded on the first search request, which may take some time.

Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

### 6. Fine-Tuning (Optional)

This repository includes placeholder scripts for fine-tuning the models:
- `scripts/fine_tune_embedder.py`: For contrastive fine-tuning of the `all-MiniLM-L6-v2` sentence embedder.
- `scripts/fine_tune_flan_lora.py`: For instruction fine-tuning the `Flan-T5-Small` model using LoRA.

The implementation of these scripts is planned for future work.
