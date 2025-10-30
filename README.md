# SLM for Academic Calendar Interpretation

This project is a lightweight pipeline that ingests academic calendar PDFs and images, extracts structured events, and answers natural language queries about them. The entire process runs locally without relying on external LLM APIs.

This implementation now includes a dual-stage Small Language Model (SLM) pipeline for improved semantic search and natural language answers, alongside the original TF-IDF baseline.

## Features

- **Data Ingestion:** Supports both PDF and image files.
- **Text Extraction:** Uses PyMuPDF for PDFs and Tesseract OCR for images.
- **Structured Data:** Parses raw text into a structured CSV format.
- **Date Normalization:** Normalizes various date formats into ISO 8601.
- **Two QA Systems:**
    1.  **Baseline:** A simple TF-IDF-based retrieval model.
    2.  **SLM Pipeline:** A semantic search system using `all-MiniLM-L6-v2` embeddings with a FAISS index for retrieval, and `Flan-T5-Small` for generating natural language answers.
- **Web Interface:** A Flask-based UI to ask questions.

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
│   └── baseline_retrieval.py # Builds the search index (TF-IDF or FAISS)
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

First, run the initial data processing steps:
```bash
python3 run_pipeline.py
```
This will generate `data/calendar_events.csv`.

Next, build the search index. You can choose between TF-IDF and the new SLM-based FAISS index.

**Option A: Build TF-IDF Index (Baseline)**
```bash
python3 src/baseline_retrieval.py --retriever tfidf
```
This creates `data/tfidf_vectorizer.joblib` and `data/tfidf_matrix.joblib`.

**Option B: Build FAISS Index (Recommended)**
```bash
python3 src/baseline_retrieval.py --retriever slm
```
This will download the `all-MiniLM-L6-v2` model and create `data/faiss.index` and `data/embeddings.npy`.

### 5. Run the Web Application

You can run the web app with either the TF-IDF retriever or the SLM retriever.

**Option A: Run with TF-IDF Retriever**
```bash
python3 app/app.py --retriever tfidf
```

**Option B: Run with SLM Retriever (Recommended)**
```bash
python3 app/app.py --retriever slm
```
This will download the `Flan-T5-Small` model on first run.

Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

### 6. Fine-Tuning (Optional)

This repository includes placeholder scripts for fine-tuning the models:
- `scripts/fine_tune_embedder.py`: For contrastive fine-tuning of the `all-MiniLM-L6-v2` sentence embedder to improve retrieval accuracy on academic calendar data.
- `scripts/fine_tune_flan_lora.py`: For instruction fine-tuning the `Flan-T5-Small` model using LoRA to improve the quality of generated answers.

The implementation of these scripts is planned for future work.
