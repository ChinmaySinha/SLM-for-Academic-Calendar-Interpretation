from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

app = Flask(__name__)

# Load the data and the TF-IDF model
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
try:
    df = pd.read_csv(os.path.join(DATA_PATH, 'calendar_events.csv'))
    vectorizer = joblib.load(os.path.join(DATA_PATH, 'tfidf_vectorizer.joblib'))
    tfidf_matrix = joblib.load(os.path.join(DATA_PATH, 'tfidf_matrix.joblib'))
    print(f"Loaded {len(df)} events from calendar_events.csv")
except FileNotFoundError as e:
    print(f"Error: Model files not found - {e}")
    print("Please run the data processing pipeline first.")
    df = pd.DataFrame()
    vectorizer = None
    tfidf_matrix = None

def enhance_query(query):
    """Enhance the query with synonyms and related terms."""
    query_lower = query.lower()
    enhancements = []
    
    # Add synonyms and related terms
    if 'drop' in query_lower or 'withdraw' in query_lower:
        enhancements.extend(['add/drop', 'withdraw', 'course drop', 'course withdraw'])
    
    if 'exam' in query_lower or 'test' in query_lower:
        enhancements.extend(['assessment test', 'cat', 'examination', 'final assessment', 'fat'])
    
    if 'register' in query_lower or 'registration' in query_lower:
        enhancements.extend(['registration', 'course registration', 'register'])
    
    if 'holiday' in query_lower or 'vacation' in query_lower:
        enhancements.extend(['holiday', 'vacation', 'no instructional'])
    
    if 'last day' in query_lower or 'last date' in query_lower:
        enhancements.extend(['last instructional day', 'last date', 'final day'])
    
    if 'semester' in query_lower:
        if 'start' in query_lower or 'begin' in query_lower or 'commence' in query_lower:
            enhancements.extend(['commencement', 'semester starts', 'first instructional'])
        if 'end' in query_lower or 'finish' in query_lower:
            enhancements.extend(['last instructional', 'semester ends'])
    
    if 'fee' in query_lower or 'payment' in query_lower:
        enhancements.extend(['fee payment', 'payment', 're-registration fees'])
    
    # Combine original query with enhancements
    enhanced = query + ' ' + ' '.join(enhancements)
    return enhanced

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query or vectorizer is None:
        return render_template('index.html', results=[])
    
    # Enhance the query
    enhanced_query = enhance_query(query)
    
    # Transform the query using the loaded vectorizer
    query_vector = vectorizer.transform([enhanced_query])
    
    # Calculate cosine similarity between the query and the calendar events
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the top 10 most similar events
    top_indices = cosine_similarities.argsort()[-10:][::-1]
    
    results = []
    for i in top_indices:
        # Lower threshold to catch more results
        if cosine_similarities[i] > 0.01:
            event = df.iloc[i].to_dict()
            event['score'] = round(cosine_similarities[i], 4)
            results.append(event)
    
    # If no results with current threshold, return top 5 regardless
    if not results and len(top_indices) > 0:
        for i in top_indices[:5]:
            event = df.iloc[i].to_dict()
            event['score'] = round(cosine_similarities[i], 4)
            results.append(event)
    
    print(f"Query: '{query}' -> Found {len(results)} results")
    
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True, port=5000)