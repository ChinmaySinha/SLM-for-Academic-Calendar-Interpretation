import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def build_and_save_model(data_path='data/calendar_processed.csv', vectorizer_path='data/vectorizer.joblib', matrix_path='data/tfidf_matrix.joblib'):
    """
    Builds the TF-IDF model and saves it to disk.
    """
    # Load the processed data
    df = pd.read_csv(data_path)

    # Create a combined text field for indexing
    df['combined_text'] = df['details_text'].fillna('') + ' ' + \
                          df['day_text'].fillna('') + ' ' + \
                          df['event_type'].fillna('')

    # Initialize and train the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    # Save the vectorizer and the matrix
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)

    print("TF-IDF model built and saved successfully.")
    return df, vectorizer, tfidf_matrix

def search_events(query, df, vectorizer, tfidf_matrix, top_n=3):
    """
    Searches for the most relevant events for a given query.
    """
    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query and all events
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the top N most similar events
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Return the top N events
    return df.iloc[top_indices]

if __name__ == '__main__':
    # Build the model
    df, vectorizer, tfidf_matrix = build_and_save_model()

    # Example usage
    print("\n--- Testing the search function ---")
    test_query = "When is the last day to drop a class?"
    results = search_events(test_query, df, vectorizer, tfidf_matrix)

    print(f"Query: '{test_query}'")
    print("Results:")
    print(results[['raw_date_text', 'details_text', 'normalized_date_start']])
