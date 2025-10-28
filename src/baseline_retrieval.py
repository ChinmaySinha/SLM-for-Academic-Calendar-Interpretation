import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def main():
    input_path = 'data/calendar_events.csv'
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} events from {input_path}")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run the full data processing pipeline first.")
        return
    
    # Fill NaN values in all text columns with an empty string
    df['details_text'] = df['details_text'].fillna('')
    df['raw_date_text'] = df['raw_date_text'].fillna('')
    df['day_text'] = df['day_text'].fillna('')
    
    # Create a combined text column for indexing with weighted importance
    # Give more weight to details text as it's the most important
    df['combined_text'] = (
        df['details_text'] + ' ' + 
        df['details_text'] + ' ' +  # Repeat details for more weight
        df['raw_date_text'] + ' ' + 
        df['day_text']
    )
    
    # Create and train the TF-IDF vectorizer with custom parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
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
    
    # Print sample features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nSample features: {feature_names[:20]}")

if __name__ == '__main__':
    main()