import os
import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re

def clean_text(text):
    """Removes OCR noise, extra whitespace, and common artifacts."""
    if not isinstance(text, str):
        return ""

    # 1. Remove specific OCR noise patterns observed (like Sas*, SE lees omen etc.)
    #    Regex to remove patterns like '(ES lana', 'SE (lees omen', etc.
    text = re.sub(r'\([A-Z]{2}\s+\w+\)?', '', text) # Removes '(ES lana)' etc.
    text = re.sub(r'[A-Z]{2}\s+\(.*\)?', '', text) # Removes 'SE (lees omen)' etc.
    text = re.sub(r'Sas\*', '', text) # Removes Sas*
    text = re.sub(r'Fe\s?a?', '', text) # Removes 'Fe a' or 'Fea'
    text = re.sub(r'R\d{1,2}\s?\[ES', '', text) # Removes R82 [ES etc.
    text = re.sub(r'\[\w+\]', '', text) # Removes bracketed noise like [istsa0zt]
    text = re.sub(r'\b[A-Za-z]{1,2}\b(?=\s+\()', '', text) # Removes 1-2 letter words before parens

    # 2. Remove leading/trailing quotes, pipes, equals, hyphens, asterisks
    text = text.strip(' "\'|=*-_')

    # 3. Replace problematic characters
    text = text.replace("‘", "'").replace("’", "'").replace("|", " ")

    # 4. Standardize whitespace (collapse multiple spaces/newlines to one space)
    text = ' '.join(text.split())

    # 5. Remove trailing connectors or junk
    text = text.rstrip(' .,:;-+=*&_')

    # 6. Specific cleanup for known messy text if needed (add more rules here)
    text = text.replace('adddrop /option', 'add/drop option')
    text = text.replace('(Holida', '(Holiday)') # Fix common OCR typo

    return text.strip() # Final strip

# --- (extract_text_from_image and extract_text_from_pdf remain the same as previous version) ---
def extract_text_from_image(file_path):
    """Extracts text from an image file using Tesseract OCR."""
    try:
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(Image.open(file_path), config=custom_config)
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file by rendering each page as an image and performing OCR."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            custom_config = r'--oem 3 --psm 6'
            text += pytesseract.image_to_string(img, config=custom_config) + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return ""

# --- (parse_raw_text and main remain the same as previous version) ---
def parse_raw_text(text, source_file):
    """
    Parses OCR text using a state machine, optimized for table images
    with multi-line cells. (Same logic as v8)
    """
    events = []
    current_event = None
    date_pattern_start = re.compile(
        r'^(\d{2}[.-]\d{2}[.-]\d{4}(\s*(?:to|&)\s*\d{2}[.-]\d{2}[.-]\d{4})?)'
    )
    date_pattern_only = re.compile(r'^\d{2}[.-]\d{2}[.-]\d{4}$')
    day_patterns = sorted([
        "Monday & Tuesday", "Monday to Wednesday", "Tuesday to Monday",
        "Thursday to Sunday", "Friday to Sunday", "Sunday to Monday",
        "Sunday to Sunday", "Wednesday to Sunday", "Saturday to Friday",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "Saturda", "Wednesda", "Frida" # Handle common OCR errors
    ], key=len, reverse=True)
    day_map = {"Saturda": "Saturday", "Wednesda": "Wednesday", "Frida": "Friday"}
    footer_strings_start = [
        'note:', 'students should participate', 'm. anthony xavior', 'dean academics',
        'no. of instructional days', 'only re-registration', 'page 1 of 1',
        'timed-out students', 'cat-i', 'actual schedule',
        'it is necessary to maintain', 'a minimum of 75%', 'last date for the upload',
        'fat schedule will be announced', 'students are required to contact'
    ]
    junk_strings_exact = ['date', 'day', 'details', 'activity', 'vit', '&', 'to']
    junk_strings_start = [
        'vit/vlr/acad', 'circular', 'vellore institute of',
        'office of the dean', '(academics)',
        'date day details', 'date(s) day activity', 'applicable to the students',
        'phd', 'india', 'vellore-', 'nadu',
        'except bsc.', 'except mba', 'deemed to be university', 'ugc act',
        'the following table:'
    ]
    lines = text.strip().split('\n')

    for line in lines:
        # Apply cleaning *before* parsing logic
        cleaned_line = clean_text(line)
        if not cleaned_line: continue
        line_lower = cleaned_line.lower()

        if any(line_lower.startswith(footer) for footer in footer_strings_start):
            if current_event: events.append(current_event)
            current_event = None
            break

        if line_lower in junk_strings_exact or any(line_lower.startswith(junk) for junk in junk_strings_start):
             continue

        date_match = date_pattern_start.match(cleaned_line)
        is_new_event_line = bool(date_match)

        if is_new_event_line and current_event:
            remaining_text_after_date = cleaned_line[date_match.end(1):].strip().lower()
            if remaining_text_after_date == 'to' or remaining_text_after_date == '&':
                if current_event['raw_date_text'].endswith(' to') or current_event['raw_date_text'].endswith(' &'):
                     current_event['raw_date_text'] += " " + date_match.group(1).replace('-', '.')
                     if current_event['details_text'].lower() == 'to' or current_event['details_text'] == '&':
                         current_event['details_text'] = ""
                     is_new_event_line = False

        if is_new_event_line:
            if current_event: events.append(current_event)
            raw_date = date_match.group(1).replace('-', '.')
            remaining_text = cleaned_line[date_match.end(1):].strip()
            day = ""
            details = remaining_text
            current_remaining = remaining_text
            for pattern in day_patterns:
                if current_remaining.lower().lstrip().startswith(pattern.lower()):
                    day_pattern_actual = current_remaining.lstrip()[:len(pattern)]
                    potential_details = current_remaining.lstrip()[len(pattern):].strip()
                    if len(potential_details) > 1 or not potential_details:
                        day = day_pattern_actual
                        details = potential_details
                        break
            day = day_map.get(day, day)
            current_event = {
                'raw_date_text': raw_date,
                'day_text': day,
                'details_text': details,
                'source_file': source_file
            }
        elif current_event:
            ends_with_connector = current_event['raw_date_text'].endswith(' to') or current_event['raw_date_text'].endswith(' &')
            is_date_only_line = date_pattern_only.match(cleaned_line)
            if ends_with_connector and is_date_only_line:
                 current_event['raw_date_text'] += " " + cleaned_line.replace('-', '.')
                 if current_event['details_text'].lower() == 'to' or current_event['details_text'] == '&':
                     current_event['details_text'] = ""
            elif not current_event.get('day_text'):
                is_day_part = False
                for pattern in day_patterns:
                   if line_lower.startswith(pattern.lower()):
                        text_after_day = cleaned_line[len(pattern):].strip()
                        if len(text_after_day) > 5 and not date_pattern_find.search(text_after_day):
                             current_event['day_text'] = cleaned_line[:len(pattern)]
                             # Apply clean_text to the details part as well
                             current_event['details_text'] = clean_text(text_after_day)
                        else:
                             current_event['day_text'] = cleaned_line
                        is_day_part = True
                        break
                if not is_day_part and (current_event['details_text'] or cleaned_line) :
                     current_event['details_text'] += " " + cleaned_line
            elif current_event['details_text'] or cleaned_line:
                 current_event['details_text'] += " " + cleaned_line

    if current_event: events.append(current_event)

    if not events: return pd.DataFrame()
    df = pd.DataFrame(events)

    # Apply final cleaning after DataFrame creation
    df['raw_date_text'] = df['raw_date_text'].apply(lambda x: clean_text(x).replace(' .', '.'))
    df['day_text'] = df['day_text'].apply(clean_text)
    df['details_text'] = df['details_text'].apply(clean_text) # Apply enhanced cleaning

    # Refined duplicate and junk removal
    df = df.drop_duplicates(subset=['raw_date_text', 'details_text'], keep='first')
    df = df[~(df['details_text'].str.fullmatch(r'&|to', na=False, flags=re.IGNORECASE))]
    df = df[~(df['details_text'].str.len() < 3 & df['day_text'].isna()) | (df['details_text'] == "")] # Keep if details are empty but day exists

    return df

def main():
    data_dir = 'data'
    all_events_dfs = []

    # Process only image files if specified, otherwise process pdfs too
    allowed_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    filenames = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(allowed_extensions)])

    if not filenames:
        print(f"No suitable files found in {data_dir}.")
        return

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        raw_text = ""
        print(f"Processing file: {filename}")
        # Decide whether to use PDF or Image extraction based on extension
        if filename.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path) # Keep PDF processing
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
             raw_text = extract_text_from_image(file_path)

        if raw_text:
            events_df = parse_raw_text(raw_text, filename)
            if not events_df.empty:
                all_events_dfs.append(events_df)

    if not all_events_dfs:
        print("No events extracted from the files.")
        return

    df = pd.concat(all_events_dfs, ignore_index=True)

    # Apply final cleaning *after* combining all data
    df['raw_date_text'] = df['raw_date_text'].apply(lambda x: clean_text(x).replace(' .', '.'))
    df['day_text'] = df['day_text'].apply(clean_text)
    df['details_text'] = df['details_text'].apply(clean_text)

    df = df.drop_duplicates(subset=['raw_date_text', 'details_text'], keep='first')

    df['id'] = range(1, len(df) + 1)
    df = df[['id', 'raw_date_text', 'day_text', 'details_text', 'source_file']]

    output_path = os.path.join(data_dir, 'calendar_events.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path} with {len(df)} events.")

if __name__ == '__main__':
    main()