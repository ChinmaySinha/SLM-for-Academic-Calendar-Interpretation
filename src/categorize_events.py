import pandas as pd

def categorize_event(details_text):
    """Categorizes an event based on its details text."""
    details_lower = details_text.lower()

    if 'add/drop' in details_lower or 'course withdraw' in details_lower:
        return 'add_drop_withdraw'
    if 'registration' in details_lower:
        return 'registration'
    if 'semester starts' in details_lower or 'commencement of classes' in details_lower:
        return 'semester_start'
    if 'semester ends' in details_lower:
        return 'semester_end'
    if 'mid semester examination' in details_lower:
        return 'exam'
    if 'end semester examination' in details_lower:
        return 'exam'
    if 're-examination' in details_lower:
        return 'exam'
    if 'holiday' in details_lower or 'vacation' in details_lower:
        return 'holiday_vacation'
    if 'fee payment' in details_lower:
        return 'fee_payment'

    return 'other'

def main():
    input_path = 'data/calendar_events.csv'
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run the ingestion and normalization scripts first.")
        return

    df['event_type'] = df['details_text'].apply(categorize_event)

    df.to_csv(input_path, index=False)
    print(f"Successfully categorized events in {input_path}")

if __name__ == '__main__':
    main()
