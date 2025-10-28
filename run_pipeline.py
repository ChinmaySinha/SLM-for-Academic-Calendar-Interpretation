"""
Simplified pipeline runner that directly imports and runs each script.
This avoids subprocess encoding issues on Windows.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_step(step_name, module_name):
    """Run a pipeline step by importing and executing it."""
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print('='*70)
    
    try:
        # Import and run the module's main function
        if module_name == 'ingest_data':
            from src import ingest_data
            ingest_data.main()
        elif module_name == 'normalize_dates':
            from src import normalize_dates
            normalize_dates.main()
        elif module_name == 'categorize_events':
            from src import categorize_events
            categorize_events.main()
        elif module_name == 'baseline_retrieval':
            from src import baseline_retrieval
            baseline_retrieval.main()
        
        print(f"\n[OK] Completed {step_name}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed at {step_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    print("="*70)
    print("ACADEMIC CALENDAR DATA PROCESSING PIPELINE")
    print("="*70)
    
    steps = [
        ("Data Ingestion (OCR & Parsing)", "ingest_data"),
        ("Date Normalization", "normalize_dates"),
        ("Event Categorization", "categorize_events"),
        ("Search Index Building", "baseline_retrieval")
    ]
    
    for step_name, module_name in steps:
        run_step(step_name, module_name)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\nGenerated files:")
    print("  - data/calendar_events.csv")
    print("  - data/tfidf_vectorizer.joblib")
    print("  - data/tfidf_matrix.joblib")
    
    print("\nNext steps:")
    print("  1. Verify extraction: python test_extraction.py")
    print("  2. Start web app: python app/app.py")
    print("  3. Open browser: http://127.0.0.1:5000")

if __name__ == '__main__':
    main()