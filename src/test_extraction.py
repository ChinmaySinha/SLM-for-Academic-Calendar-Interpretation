#!/usr/bin/env python3
"""
Test script to verify the accuracy of calendar extraction.
Compares extracted events against expected events from the image.
"""

import pandas as pd
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Expected events from the calendar image you showed
EXPECTED_EVENTS = [
    ("06.10.2023 & 07.10.2023", "Course wish list registration by students"),
    ("09.10.2023 to 20.10.2023", "Course allocation and scheduling by Schools"),
    ("26.10.2023", "Mock - Course registration for Freshers"),
    ("29.10.2023", "Course registration by students"),
    ("03.01.2024", "Commencement of Winter Semester 2023-24"),
    ("03.01.2024 to 05.01.2024", "Course add/drop option to students"),
    ("12.01.2024", "Last date for the payment of re-registration fees"),
    ("13.01.2024 to 15.01.2024", "Pongal Holidays"),
    ("11.02.2024 to 17.02.2024", "Continuous Assessment Test -1"),
    ("29.02.2024 to 03.03.2024", "Riviera 2024"),
    ("04.03.2024 to 06.03.2024", "Course withdraw option for students"),
    ("29.03.2024", "Good Friday (No Instructional Day)"),
    ("01.04.2024 to 07.04.2024", "Continuous Assessment Test - II"),
    ("09.04.2024", "Telugu New Year's Day (No Instructional Day)"),
    ("14.04.2024", "Tamil New Year's Day/ Dr. B. R. Ambedkar Birthday (Holiday)"),
    ("25.04.2024 & 26.04.2024", "25th Science Engineering and Technology (SET) Conference"),
    ("26.04.2024", "Last instructional day for laboratory classes"),
    ("29.04.2024 to 03.05.2024", "Final assessment test for laboratory courses/ components"),
    ("03.05.2024", "Last instructional day for theory classes"),
    ("06.05.2024", "Commencement of final assessment test for theory courses / components"),
    ("20.05.2024", "Commencement of Summer Term 2023-24 (Tentative)"),
]

def test_extraction():
    """Test the extraction accuracy."""
    
    csv_path = 'data/calendar_events.csv'
    
    if not os.path.exists(csv_path):
        print("[ERROR] calendar_events.csv not found!")
        print("Please run: python3 run_pipeline.py")
        return
    
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("CALENDAR EXTRACTION TEST REPORT")
    print("=" * 70)
    print(f"\n[INFO] Total events extracted: {len(df)}")
    print(f"[INFO] Expected events: {len(EXPECTED_EVENTS)}")
    
    # Check extraction rate
    extraction_rate = (len(df) / len(EXPECTED_EVENTS)) * 100
    print(f"[INFO] Extraction rate: {extraction_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("EXTRACTED EVENTS")
    print("=" * 70)
    
    # Display all extracted events
    for idx, row in df.iterrows():
        print(f"\n{idx + 1}. Date: {row['raw_date_text']}")
        print(f"   Day: {row['day_text']}")
        print(f"   Details: {row['details_text']}")
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    # Check for key events
    key_events = [
        "Course wish list registration",
        "Course registration by students",
        "add/drop",
        "Continuous Assessment Test",
        "Riviera",
        "withdraw option",
        "Final assessment test",
        "Last instructional day",
    ]
    
    found_events = []
    missing_events = []
    
    for key in key_events:
        found = df['details_text'].str.contains(key, case=False, na=False).any()
        if found:
            found_events.append(key)
            print(f"[OK] Found: {key}")
        else:
            missing_events.append(key)
            print(f"[MISSING] {key}")
    
    print(f"\n[INFO] Key Events Found: {len(found_events)}/{len(key_events)}")
    
    # Check date formats
    print("\n" + "=" * 70)
    print("DATE FORMAT CHECK")
    print("=" * 70)
    
    invalid_dates = []
    for idx, row in df.iterrows():
        date_text = row['raw_date_text']
        if pd.isna(date_text) or not isinstance(date_text, str):
            invalid_dates.append((idx, date_text))
        elif not any(c.isdigit() for c in date_text):
            invalid_dates.append((idx, date_text))
    
    if invalid_dates:
        print(f"[WARN] Found {len(invalid_dates)} invalid date formats:")
        for idx, date in invalid_dates[:5]:
            print(f"   Row {idx}: {date}")
    else:
        print("[OK] All date formats look valid")
    
    # Check for empty details
    empty_details = df[df['details_text'].str.strip().str.len() < 3]
    if len(empty_details) > 0:
        print(f"\n[WARN] Found {len(empty_details)} events with empty/short details")
    else:
        print("\n[OK] All events have proper details")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    score = (len(found_events) / len(key_events)) * 100
    
    if score >= 90:
        status = "[EXCELLENT]"
    elif score >= 70:
        status = "[GOOD]"
    elif score >= 50:
        status = "[FAIR]"
    else:
        status = "[NEEDS IMPROVEMENT]"
    
    print(f"\nExtraction Quality: {status} ({score:.1f}%)")
    print(f"Total Events: {len(df)}")
    print(f"Key Events Captured: {len(found_events)}/{len(key_events)}")
    
    if missing_events:
        print(f"\n[WARN] Consider re-running with better image quality for:")
        for event in missing_events[:3]:
            print(f"   - {event}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    test_extraction()