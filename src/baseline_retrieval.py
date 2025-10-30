import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def build_tfidf_index(df):
    """Builds and saves a TF-IDF index."""
    print("Building TF-IDF index...")
    # Create and train the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_features=5000,
        min_df=1,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    # Save the vectorizer and the TF-IDF matrix
    joblib.dump(vectorizer, 'data/tfidf_vectorizer.joblib')
    joblib.dump(tfidf_matrix, 'data/tfidf_matrix.joblib')

    print("Successfully built and saved the TF-IDF model.")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

def build_slm_index(df):
    """Builds and saves a semantic (FAISS) index."""
    print("Building SLM index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    print("Generating embeddings for calendar events...")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # Dimension of embeddings
    d = embeddings.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))

    # Save the index and embeddings
    faiss.write_index(index, 'data/faiss.index')
    np.save('data/embeddings.npy', embeddings)

    print("Successfully built and saved the FAISS index and embeddings.")
    print(f"FAISS index size: {index.ntotal} vectors")
    print(f"Embeddings array shape: {embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description="Build a search index for the academic calendar.")
    parser.add_argument('--retriever', type=str, default='tfidf', choices=['tfidf', 'slm'],
                        help="The type of retrieval model to build ('tfidf' or 'slm').")
    args = parser.parse_args()

    input_path = 'data/calendar_events.csv'
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} events from {input_path}")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run the full data processing pipeline first.")
        return
    
    # Fill NaN values and create a combined text column
    df['details_text'] = df['details_text'].fillna('')
    df['raw_date_text'] = df['raw_date_text'].fillna('')
    df['day_text'] = df['day_text'].fillna('')
    df['combined_text'] = (
        df['details_text'] + ' ' + 
        df['details_text'] + ' ' +
        df['raw_date_text'] + ' ' + 
        df['day_text']
    )
    
    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    if args.retriever == 'tfidf':
        build_tfidf_index(df)
    elif args.retriever == 'slm':
        build_slm_index(df)

if __name__ == '__main__':
    main()
