import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def main():
    """Builds and saves a semantic (FAISS) index for the academic calendar."""
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
        df['details_text'] + ' ' +  # Weight details more
        df['raw_date_text'] + ' ' + 
        df['day_text']
    )
    
    print("Building SLM index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings for calendar events...")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))

    if not os.path.exists('data'):
        os.makedirs('data')

    faiss.write_index(index, 'data/faiss.index')
    np.save('data/embeddings.npy', embeddings)

    print("Successfully built and saved the FAISS index and embeddings.")
    print(f"FAISS index size: {index.ntotal} vectors")
    print(f"Embeddings array shape: {embeddings.shape}")

if __name__ == '__main__':
    main()
