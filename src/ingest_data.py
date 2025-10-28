import os
import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re

def extract_text_from_image(file_path):
    """Extracts text from an image file using Tesseract OCR."""
    try:
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return ""

def parse_raw_text(text, source_file):
    """Parses raw text to extract calendar events with improved robustness."""
    lines = text.strip().split('\n')
    events = []
    current_event = None

    # Regex to capture dates (dd.mm.yyyy), date ranges (dd.mm.yyyy to dd.mm.yyyy), and days
    date_pattern = re.compile(
        r'^(?P<date>\d{2}\.\d{2}\.\d{4}(?:\s*to\s*\d{2}\.\d{2}\.\d{4})?)\s+'
        r'(?P<day>[A-Za-z\s]+?)\s+'
        r'(?P<details>.*)$'
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = date_pattern.match(line)
        if match:
            # If a new date pattern is found, save the previous event
            if current_event:
                events.append(current_event)

            data = match.groupdict()
            current_event = {
                'raw_date_text': data['date'].strip(),
                'day_text': data['day'].strip(),
                'details_text': data['details'].strip(),
                'source_file': source_file
            }
        elif current_event:
            # If it's not a new event, append the line to the details of the current event
            current_event['details_text'] += ' ' + line

    # Append the last event
    if current_event:
        events.append(current_event)

    return events

def main():
    data_dir = 'data'
    all_events = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        raw_text = ""

        if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            print(f"Processing file: {filename}")
            if filename.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(file_path)
            else:
                raw_text = extract_text_from_image(file_path)

        if raw_text:
            events = parse_raw_text(raw_text, filename)
            all_events.extend(events)

    if not all_events:
        print("No events extracted. Please check the files in the 'data' directory.")
        return

    df = pd.DataFrame(all_events)
    df['id'] = range(1, len(df) + 1)

    df = df[['id', 'raw_date_text', 'day_text', 'details_text', 'source_file']]

    output_path = os.path.join(data_dir, 'calendar_events.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path} with {len(df)} events.")

if __name__ == '__main__':
    main()
