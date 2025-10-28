from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load the data and the TF-IDF model
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
try:
    df = pd.read_csv(os.path.join(DATA_PATH, 'calendar_events.csv'))
    vectorizer = joblib.load(os.path.join(DATA_PATH, 'tfidf_vectorizer.joblib'))
    tfidf_matrix = joblib.load(os.path.join(DATA_PATH, 'tfidf_matrix.joblib'))
except FileNotFoundError:
    print("Error: Model files not found. Please run the data processing pipeline first.")
    df = pd.DataFrame() # Empty dataframe to avoid errors on startup
    vectorizer = None
    tfidf_matrix = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query or vectorizer is None:
        return render_template('index.html', results=[])

    # Transform the query using the loaded vectorizer
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query and the calendar events
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the top 5 most similar events
    top_indices = cosine_similarities.argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        # --- START OF FIX ---
        # Lowered the threshold to 0.01 to catch more relevant results
        if cosine_similarities[i] > 0.01:
        # --- END OF FIX ---
             results.append(df.iloc[i].to_dict())

    return render_template('index.html', results=results, query=query)
if __name__ == '__main__':
    app.run(debug=True)
