from flask import Flask, render_template, request
import pandas as pd
import joblib
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from baseline_retrieval import search_events

app = Flask(__name__)

# Load the model and data
try:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'calendar_processed.csv')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vectorizer.joblib')
    matrix_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tfidf_matrix.joblib')

    df = pd.read_csv(data_path)
    df['combined_text'] = df['details_text'].fillna('') + ' ' + \
                          df['day_text'].fillna('') + ' ' + \
                          df['event_type'].fillna('')

    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)
except FileNotFoundError as e:
    print(f"Error loading model or data: {e}")
    print("Please ensure the model is built by running 'src/baseline_retrieval.py' first.")
    df, vectorizer, tfidf_matrix = None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    query = ""
    if request.method == 'POST':
        query = request.form['query']
        if df is not None:
            results_df = search_events(query, df, vectorizer, tfidf_matrix)
            results = results_df.to_dict('records')

    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5001) # Use a different port than default
