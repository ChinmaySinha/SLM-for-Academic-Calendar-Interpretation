import csv
import dateparser
from datetime import datetime

def parse_date_text(raw_text):
    """
    Parses a raw date string and returns a start and end date.
    Handles single dates, ranges with "to", and multiple dates with "&".
    """
    raw_text = raw_text.strip()
    start_date, end_date = None, None

    try:
        if 'to' in raw_text:
            parts = raw_text.split('to')
            start_str = parts[0].strip()
            end_str = parts[1].strip()

            # Help dateparser by providing the format
            start_date = dateparser.parse(start_str, date_formats=['%d.%m.%Y'])
            end_date = dateparser.parse(end_str, date_formats=['%d.%m.%Y'])

        elif '&' in raw_text:
            parts = raw_text.split('&')
            start_str = parts[0].strip()
            end_str = parts[1].strip()

            start_date = dateparser.parse(start_str, date_formats=['%d.%m.%Y'])
            end_date = dateparser.parse(end_str, date_formats=['%d.%m.%Y'])

        else:
            # Single date
            start_date = dateparser.parse(raw_text, date_formats=['%d.%m.%Y'])
            end_date = start_date

    except Exception as e:
        print(f"Could not parse date: {raw_text}. Error: {e}")
        return None, None

    # Format to ISO 8601 (YYYY-MM-DD)
    start_iso = start_date.strftime('%Y-%m-%d') if start_date else None
    end_iso = end_date.strftime('%Y-%m-%d') if end_date else None

    return start_iso, end_iso


def main():
    """
    Reads the parsed CSV, normalizes dates, and writes to a new CSV.
    """
    input_filepath = 'data/calendar.csv'
    output_filepath = 'data/calendar_normalized.csv'

    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['normalized_date_start', 'normalized_date_end']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            raw_date_text = row.get('raw_date_text', '').strip()
            if raw_date_text:
                start_date, end_date = parse_date_text(raw_date_text)
                row['normalized_date_start'] = start_date
                row['normalized_date_end'] = end_date
            else:
                row['normalized_date_start'] = None
                row['normalized_date_end'] = None

            writer.writerow(row)

if __name__ == '__main__':
    main()
