import csv

def categorize_event(details_text):
    """
    Categorizes an event based on keywords in its details.
    """
    details_lower = details_text.lower()

    # Synonym mapping
    if any(s in details_lower for s in ["drop course", "withdraw", "course withdraw"]):
        return "withdrawal"
    if "re-registration" in details_lower:
        return "registration"

    # Rule-based categorization
    if "vacation" in details_lower:
        return "vacation"
    if "holiday" in details_lower:
        return "holiday"
    if "registration" in details_lower:
        return "registration"
    if "add/drop" in details_lower:
        return "add_drop"
    if "commencement" in details_lower:
        return "semester_event"
    if "assessment test" in details_lower:
        return "exam"
    if "instructional day" in details_lower:
        return "instructional_day"
    if "riviera" in details_lower:
        return "event"

    return "other"

def main():
    """
    Reads the normalized CSV, categorizes events, and writes to a new CSV.
    """
    input_filepath = 'data/calendar_normalized.csv'
    output_filepath = 'data/calendar_processed.csv'

    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['event_type']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            details = row.get('details_text', '')
            event_type = categorize_event(details)
            row['event_type'] = event_type
            writer.writerow(row)

if __name__ == '__main__':
    main()
