import pandas as pd
import dateparser
from datetime import timedelta

def normalize_date_text(date_text):
    """Normalizes a single date string or a date range string."""
    parser_settings = {'DATE_ORDER': 'DMY'}

    # Handle date ranges separated by "to"
    if 'to' in date_text:
        start_str, end_str = date_text.split('to')
        start_date = dateparser.parse(start_str.strip(), settings=parser_settings)
        end_date = dateparser.parse(end_str.strip(), settings=parser_settings)
        return start_date, end_date

    # Handle multiple dates separated by "&"
    if '&' in date_text:
        dates_str = date_text.split('&')
        start_date = dateparser.parse(dates_str[0].strip(), settings=parser_settings)
        end_date = dateparser.parse(dates_str[-1].strip(), settings=parser_settings)
        return start_date, end_date

    # Handle single dates
    date = dateparser.parse(date_text.strip(), settings=parser_settings)
    return date, date

def main():
    input_path = 'data/calendar_events.csv'
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run the ingestion script first.")
        return

    normalized_dates = []
    for index, row in df.iterrows():
        start_date, end_date = normalize_date_text(row['raw_date_text'])
        normalized_dates.append({
            'normalized_date_start': start_date.strftime('%Y-%m-%d') if start_date else None,
            'normalized_date_end': end_date.strftime('%Y-%m-%d') if end_date else None
        })

    dates_df = pd.DataFrame(normalized_dates)

    # Update the original dataframe with the new columns
    df['normalized_date_start'] = dates_df['normalized_date_start']
    df['normalized_date_end'] = dates_df['normalized_date_end']

    df.to_csv(input_path, index=False)
    print(f"Successfully normalized dates in {input_path}")

if __name__ == '__main__':
    main()
