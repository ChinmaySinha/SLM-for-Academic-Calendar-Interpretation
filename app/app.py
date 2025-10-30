from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Global Variables ---
RETRIEVER_TYPE = 'tfidf'
df = None
# TF-IDF specific
vectorizer = None
tfidf_matrix = None
# SLM specific
retrieval_model = None
faiss_index = None
generation_model = None
generation_tokenizer = None

# --- Model Loading ---
def load_models(retriever_type):
    """Load all necessary models based on the retriever type."""
    global df, RETRIEVER_TYPE
    global vectorizer, tfidf_matrix
    global retrieval_model, faiss_index, generation_model, generation_tokenizer
    
    RETRIEVER_TYPE = retriever_type
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, 'calendar_events.csv'))
        print(f"Loaded {len(df)} events from calendar_events.csv")
    except FileNotFoundError:
        print("Error: calendar_events.csv not found. Please run the data processing pipeline first.")
        df = pd.DataFrame()
        return

    if RETRIEVER_TYPE == 'tfidf':
        try:
            vectorizer = joblib.load(os.path.join(DATA_PATH, 'tfidf_vectorizer.joblib'))
            tfidf_matrix = joblib.load(os.path.join(DATA_PATH, 'tfidf_matrix.joblib'))
            print("Successfully loaded TF-IDF models.")
        except FileNotFoundError:
            print("Error: TF-IDF model files not found. Please run 'python src/baseline_retrieval.py --retriever tfidf'")
    
    elif RETRIEVER_TYPE == 'slm':
        try:
            # Load Sentence Transformer for retrieval
            retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Load FAISS index
            faiss_index = faiss.read_index(os.path.join(DATA_PATH, 'faiss.index'))
            # Load Flan-T5 for generation
            model_name = 'google/flan-t5-small'
            generation_tokenizer = T5Tokenizer.from_pretrained(model_name)
            generation_model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("Successfully loaded SLM models (MiniLM, FAISS, Flan-T5).")
        except Exception as e:
            print(f"Error loading SLM models: {e}")
            print("Please run 'python src/baseline_retrieval.py --retriever slm'")

# --- Search Functions ---
def search_tfidf(query):
    """Performs search using the TF-IDF model."""
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-10:][::-1]
    
    results = []
    for i in top_indices:
        if cosine_similarities[i] > 0.01:
            event = df.iloc[i].to_dict()
            event['score'] = round(cosine_similarities[i], 4)
            results.append(event)
    return results

def search_slm(query):
    """Performs search and answer synthesis using the SLM pipeline."""
    # 1. Encode query and retrieve from FAISS
    query_embedding = retrieval_model.encode([query])
    k = 5  # Number of events to retrieve
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k)
    
    retrieved_events = [df.iloc[i] for i in indices[0]]
    
    # 2. Prepare context for Flan-T5
    context = ""
    for event in retrieved_events:
        date = event.get('normalized_date', event.get('raw_date_text', ''))
        details = event.get('details_text', '')
        context += f"- Date: {date}, Event: {details}\n"

    # 3. Generate answer with Flan-T5
    prompt = f"""
    Context from the academic calendar:
    {context}
    Based on the context above, please answer the following question.
    Provide a concise answer and include the date in ISO format (YYYY-MM-DD).

    Question: {query}
    Answer:
    """

    input_ids = generation_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = generation_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    generated_answer = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_answer, retrieved_events

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', retriever=RETRIEVER_TYPE)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query or df.empty:
        return render_template('index.html', results=[], retriever=RETRIEVER_TYPE)

    if RETRIEVER_TYPE == 'tfidf':
        results = search_tfidf(query)
        return render_template('index.html', results=results, query=query, retriever=RETRIEVER_TYPE)

    elif RETRIEVER_TYPE == 'slm':
        generated_answer, source_events = search_slm(query)
        return render_template('index.html', query=query, generated_answer=generated_answer,
                               source_events=source_events, retriever=RETRIEVER_TYPE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask app with a chosen retriever.")
    parser.add_argument('--retriever', type=str, default='tfidf', choices=['tfidf', 'slm'],
                        help="The type of retrieval model to use ('tfidf' or 'slm').")
    args = parser.parse_args()

    load_models(args.retriever)
    app.run(debug=True, port=5000)
