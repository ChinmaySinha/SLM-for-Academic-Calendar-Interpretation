import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def main():
    input_path = 'data/calendar_events.csv'
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        # Corrected the variable name in the f-string from input_t_path to input_path
        print(f"Error: {input_path} not found. Please run the full data processing pipeline first.")
        return

    # --- START OF FIX ---
    # Fill NaN values in all text columns with an empty string
    # before combining them.
    df['details_text'] = df['details_text'].fillna('')
    df['raw_date_text'] = df['raw_date_text'].fillna('')
    df['day_text'] = df['day_text'].fillna('')
    # --- END OF FIX ---

    # Create a combined text column for indexing
    df['combined_text'] = df['details_text'] + ' ' + df['raw_date_text'] + ' ' + df['day_text']

    # Create and train the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    # Save the vectorizer and the TF-IDF matrix
    joblib.dump(vectorizer, 'data/tfidf_vectorizer.joblib')
    joblib.dump(tfidf_matrix, 'data/tfidf_matrix.joblib')

    print("Successfully built and saved the TF-IDF model.")

if __name__ == '__main__':
    main()