import csv
import re

def clean_text(text):
    """Removes OCR noise and extra whitespace from text."""
    return ' '.join(text.replace("‘", "").replace("’", "").replace("|", " ").strip().split())

def parse_line(line):
    """Parses a single line from the OCR output to extract date, day, and details."""
    cleaned_line = clean_text(line)

    date_pattern = r'^(\d{2}\.\d{2}\.\d{4}(?:\s*(?:to|&)\s*\d{2}\.\d{2}\.\d{4})?)'
    date_match = re.match(date_pattern, cleaned_line)

    if not date_match:
        return None, None, None

    date = date_match.group(1).strip()
    remaining = cleaned_line[len(date):].strip()

    # The day is always before the details, so let's find it
    day_patterns = sorted([
        "Monday & Tuesday", "Monday to Wednesday", "Tuesday to Monday",
        "Thursday to Sunday", "Friday to Sunday", "Sunday to Monday",
        "Sunday to Sunday", "Wednesday to Sunday", "Saturday to Friday",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "Saturda", "Wednesda", "Frida"
    ], key=len, reverse=True)

    day = ""
    details = remaining

    for pattern in day_patterns:
        if remaining.lower().startswith(pattern.lower()):
            day = pattern
            details = remaining[len(pattern):].strip()
            break

    day_map = {"Saturda": "Saturday", "Wednesda": "Wednesday", "Frida": "Friday"}
    day = day_map.get(day, day)

    return date, day, details

def main():
    """Main function to parse OCR output and generate a CSV."""
    input_filepath = 'data/ocr_output.txt'
    output_filepath = 'data/calendar.csv'

    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', newline='', encoding='utf-8') as f_out:

        writer = csv.writer(f_out)
        writer.writerow(['raw_date_text', 'day_text', 'details_text'])

        in_table = False
        previous_line_data = {}

        for line in f_in:
            if not line.strip():
                continue

            if 'Date Day Details' in line:
                in_table = True
                continue

            if 'Note:' in line:
                break

            if in_table:
                date, day, details = parse_line(line)
                if date:
                    if previous_line_data:
                        writer.writerow(previous_line_data.values())
                    previous_line_data = {"date": date, "day": day, "details": details}
                elif previous_line_data and not date:
                    # This line is a continuation of the previous one
                    previous_line_data["details"] += " " + clean_text(line)

        # Write the last line
        if previous_line_data:
            writer.writerow(previous_line_data.values())


if __name__ == '__main__':
    main()
