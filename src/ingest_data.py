import os
import sys
import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def clean_text(text):
    """Removes OCR noise, extra whitespace, and common artifacts."""
    if not isinstance(text, str):
        return ""
    
    # Remove bracketed content that contains OCR noise
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove patterns like "REE? [" or "SRE:"
    text = re.sub(r'\b[A-Z]{2,4}\?\s*', '', text)
    text = re.sub(r'\b[A-Z]{2,}\d*\s*:', '', text)
    
    # Remove repeated characters (OCR artifacts)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Remove trailing colons, pipes, question marks
    text = re.sub(r'\s*[:\|?]+\s*$', '', text)
    
    # Remove everything after ": REE" or similar patterns
    text = re.sub(r'\s*:\s*REE.*$', '', text)
    text = re.sub(r'\s*:\s*[A-Z]{3,}.*$', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' "\'|=*-_:')
    
    return text.strip()

def extract_text_from_image(file_path):
    """Extracts text from an image file using Tesseract OCR with optimized settings."""
    try:
        img = Image.open(file_path)
        
        # Enhance image for better OCR
        from PIL import ImageEnhance
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2)
        
        # Use optimized Tesseract config for tables
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        return text
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Try direct text extraction first
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            else:
                # Fall back to OCR if no text found
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                custom_config = r'--oem 3 --psm 6'
                text += pytesseract.image_to_string(img, config=custom_config) + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return ""

def parse_calendar_table(text, source_file):
    """
    Parses calendar text specifically for table format with Date, Day, Details columns.
    """
    events = []
    lines = text.strip().split('\n')
    
    # Pattern to match dates in format DD.MM.YYYY or DD.MM.YYYY & DD.MM.YYYY or DD.MM.YYYY to DD.MM.YYYY
    date_pattern = re.compile(
        r'^(\d{2}[./]\d{2}[./]\d{4}(?:\s*(?:&|to)\s*\d{2}[./]\d{2}[./]\d{4})?)',
        re.IGNORECASE
    )
    
    # Days of the week patterns (including common OCR errors)
    day_patterns = [
        r'Monday\s*&\s*Tuesday',
        r'Monday\s+to\s+Wednesday',
        r'Tuesday\s+to\s+Monday',
        r'Thursday\s+to\s+Sunday',
        r'Friday\s+to\s+Sunday',
        r'Sunday\s+to\s+Monday',
        r'Sunday\s+to\s+Sunday',
        r'Wednesday\s+to\s+Sunday',
        r'Saturday\s+to\s+Friday',
        r'Monday', r'Tuesday', r'Wednesday', r'Thursday', 
        r'Friday', r'Saturday', r'Sunday',
        r'Frida[y]?', r'Saturda[y]?', r'Wednesda[y]?'
    ]
    
    # Footer indicators to stop parsing
    footer_keywords = [
        'note:', 'students should', 'dean academics', 'no. of instructional',
        'cat-1', 'cat-ii', 'fat', 'total', 'page 1 of', 'circular'
    ]
    
    # Header keywords to skip
    header_keywords = ['date', 'day', 'details', 'vit/', 'circular', 'academic calendar']
    
    current_event = None
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Check for footer - stop parsing
        if any(keyword in line_lower for keyword in footer_keywords):
            break
        
        # Skip header lines
        if any(keyword in line_lower for keyword in header_keywords) and len(line) < 50:
            in_table = True
            continue
        
        # Check if line starts with a date
        date_match = date_pattern.match(line)
        
        if date_match:
            # Save previous event
            if current_event and current_event['details_text'].strip():
                events.append(current_event)
            
            # Extract date
            raw_date = date_match.group(1).replace('/', '.')
            remaining = line[date_match.end():].strip()
            
            # Extract day
            day = ""
            details = remaining
            
            for day_pattern in day_patterns:
                day_match = re.match(day_pattern, remaining, re.IGNORECASE)
                if day_match:
                    day = day_match.group(0)
                    details = remaining[day_match.end():].strip()
                    break
            
            # Normalize day format
            day = day.replace('Frida', 'Friday').replace('Saturda', 'Saturday').replace('Wednesda', 'Wednesday')
            
            current_event = {
                'raw_date_text': raw_date,
                'day_text': day,
                'details_text': details,
                'source_file': source_file
            }
        
        elif current_event:
            # This is a continuation line for details
            # Skip if it looks like noise
            if len(line) > 2 and not line_lower.startswith(('vit', 'page', 'circular')):
                current_event['details_text'] += ' ' + line
    
    # Add last event
    if current_event and current_event['details_text'].strip():
        events.append(current_event)
    
    if not events:
        return pd.DataFrame()
    
    df = pd.DataFrame(events)
    
    # Clean up the extracted data
    df['raw_date_text'] = df['raw_date_text'].apply(lambda x: clean_text(x))
    df['day_text'] = df['day_text'].apply(lambda x: clean_text(x))
    df['details_text'] = df['details_text'].apply(lambda x: clean_text(x))
    
    # Additional cleaning for details - remove common OCR artifacts
    def clean_details(text):
        if not isinstance(text, str):
            return ""
        
        # Remove everything after common OCR noise indicators
        # Split at patterns like ": REE", ": [", "? [", etc.
        for pattern in [r'\s*:\s*REE', r'\s*:\s*\[', r'\s*\?\s*\[', r'\s*:\s*[A-Z]{3,}']:
            if re.search(pattern, text):
                text = re.split(pattern, text)[0]
        
        # Remove trailing punctuation artifacts
        text = re.sub(r'[:\|?]+$', '', text)
        
        return text.strip()
    
    df['details_text'] = df['details_text'].apply(clean_details)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['raw_date_text', 'details_text'], keep='first')
    
    # Filter out rows with empty details
    df = df[df['details_text'].str.len() > 3]
    
    return df

def main():
    data_dir = 'data'
    all_events_dfs = []
    
    allowed_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Get all files, excluding the CSV files we generate
    all_files = os.listdir(data_dir)
    filenames = sorted([f for f in all_files 
                       if f.lower().endswith(allowed_extensions) 
                       and not f.endswith('.csv')])
    
    if not filenames:
        print(f"No suitable files found in {data_dir}.")
        return
    
    print(f"Found {len(filenames)} files to process\n")
    
    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        raw_text = ""
        print(f"Processing file: {filename}")
        
        if filename.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raw_text = extract_text_from_image(file_path)
        
        if raw_text:
            # Save raw OCR output for debugging
            debug_path = os.path.join(data_dir, f"ocr_debug_{filename}.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            events_df = parse_calendar_table(raw_text, filename)
            if not events_df.empty:
                print(f"  [OK] Extracted {len(events_df)} events from {filename}")
                all_events_dfs.append(events_df)
            else:
                print(f"  [WARN] No events extracted from {filename}")
        else:
            print(f"  [ERROR] Failed to extract text from {filename}")
    
    if not all_events_dfs:
        print("\nNo events extracted from any files.")
        return
    
    df = pd.concat(all_events_dfs, ignore_index=True)
    
    # Final cleanup
    df = df.drop_duplicates(subset=['raw_date_text', 'details_text'], keep='first')
    
    # Add ID column
    df['id'] = range(1, len(df) + 1)
    df = df[['id', 'raw_date_text', 'day_text', 'details_text', 'source_file']]
    
    output_path = os.path.join(data_dir, 'calendar_events.csv')
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Successfully created {output_path} with {len(df)} events.")
    print(f"{'='*60}")
    print(f"\nSample of extracted events:")
    print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()