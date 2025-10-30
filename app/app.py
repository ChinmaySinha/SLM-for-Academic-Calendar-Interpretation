from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import argparse
import sys
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
# SLM specific (will be loaded lazily)
retrieval_model = None
faiss_index = None
generation_model = None
generation_tokenizer = None

# --- Model Loading ---
def load_initial_data():
    """Load the initial CSV data and TF-IDF models if specified."""
    global df, vectorizer, tfidf_matrix
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

    try:
        df = pd.read_csv(os.path.join(DATA_PATH, 'calendar_events.csv'))
        print(f"Loaded {len(df)} events from calendar_events.csv")
    except FileNotFoundError:
        print("Error: calendar_events.csv not found. Please run the data processing pipeline first.")
        sys.exit(1)

    if RETRIEVER_TYPE == 'tfidf':
        try:
            vectorizer = joblib.load(os.path.join(DATA_PATH, 'tfidf_vectorizer.joblib'))
            tfidf_matrix = joblib.load(os.path.join(DATA_PATH, 'tfidf_matrix.joblib'))
            print("Successfully loaded TF-IDF models.")
        except FileNotFoundError:
            print("Error: TF-IDF model files not found. Please run 'python src/baseline_retrieval.py --retriever tfidf'")
            sys.exit(1)

def ensure_slm_models_loaded():
    """Load the SLM models if they haven't been loaded yet."""
    global retrieval_model, faiss_index, generation_model, generation_tokenizer
    if retrieval_model is None:
        DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
        try:
            print("Lazily loading SLM models...")
            retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
            faiss_index = faiss.read_index(os.path.join(DATA_PATH, 'faiss.index'))
            model_name = 'google/flan-t5-small'
            generation_tokenizer = T5Tokenizer.from_pretrained(model_name)
            generation_model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("Successfully loaded SLM models.")
        except Exception as e:
            print(f"Error loading SLM models: {e}")
            raise e

def enhance_query(query):
    """Enhance the query with synonyms and related terms."""
    query_lower = query.lower()
    enhancements = []
    if 'drop' in query_lower or 'withdraw' in query_lower:
        enhancements.extend(['add/drop', 'withdraw', 'course drop', 'course withdraw'])
    if 'exam' in query_lower or 'test' in query_lower:
        enhancements.extend(['assessment test', 'cat', 'examination', 'final assessment', 'fat'])
    if 'register' in query_lower or 'registration' in query_lower:
        enhancements.extend(['registration', 'course registration', 'register'])
    if 'holiday' in query_lower or 'vacation' in query_lower:
        enhancements.extend(['holiday', 'vacation', 'no instructional'])
    enhanced = query + ' ' + ' '.join(enhancements)
    return enhanced

# --- Search Functions ---
def search_tfidf(query):
    """Performs search using the TF-IDF model."""
    enhanced_query = enhance_query(query)
    query_vector = vectorizer.transform([enhanced_query])
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
    ensure_slm_models_loaded()

    query_embedding = retrieval_model.encode([query])
    k = 5
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k)

    retrieved_events = [df.iloc[i] for i in indices[0]]

    context = ""
    for event in retrieved_events:
        date = event.get('normalized_date', event.get('raw_date_text', ''))
        details = event.get('details_text', '')
        context += f"- Date: {date}, Event: {details}\n"

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
        try:
            generated_answer, source_events = search_slm(query)
            return render_template('index.html', query=query, generated_answer=generated_answer,
                                   source_events=source_events, retriever=RETRIEVER_TYPE)
        except Exception as e:
            return render_template('index.html', error=str(e), retriever=RETRIEVER_TYPE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask app with a chosen retriever.")
    parser.add_argument('--retriever', type=str, default='tfidf', choices=['tfidf', 'slm'],
                        help="The type of retrieval model to use ('tfidf' or 'slm').")
    args = parser.parse_args()

    RETRIEVER_TYPE = args.retriever
    load_initial_data()
    app.run(debug=True, port=5000)
