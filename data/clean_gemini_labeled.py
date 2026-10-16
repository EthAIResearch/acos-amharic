"""Clean Gemini_Labeled_ACOS_Data.csv by:
1. Retaining only Amharic texts containing valid Ethiopic script.
2. Removing English, other languages, unreadable/foreign-script texts (Arabic, Russian, etc.).
3. Removing fully duplicate rows where both text and quad (aspect_term, aspect_category,
   opinion_term, sentiment) are identical, while preserving multi-quad annotations for the same text.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

ETHIOPIC_PATTERN = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")


def clean_dataset(input_csv: str, output_csv: str):
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    total_rows = 0
    kept_rows = 0
    removed_non_amharic = 0
    removed_no_ethiopic = 0
    removed_duplicates = 0

    seen_text_quads = set()
    text_quad_counts = Counter()

    with open(input_path, "r", encoding="utf-8-sig") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        # Normalize header in case of BOM or spacing
        cleaned_header = [h.lstrip("\ufeff").strip() for h in header]
        writer.writerow(cleaned_header)

        for row in reader:
            total_rows += 1
            if len(row) != 7:
                print(f"Warning: malformed row {total_rows}: {row}")
                continue

            row_id, text, lang, aspect_term, aspect_cat, opinion_term, sentiment = row
            text_str = text.strip()
            lang_str = lang.strip()

            # 1. Retain only Amharic rows
            if lang_str != "Amharic":
                removed_non_amharic += 1
                continue

            # 2. Check for readability: must contain valid Ethiopic characters
            if not ETHIOPIC_PATTERN.search(text_str):
                removed_no_ethiopic += 1
                continue

            # 3. Deduplication: check if (text, quad) is already seen
            quad_key = (
                text_str,
                aspect_term.strip(),
                aspect_cat.strip(),
                opinion_term.strip(),
                str(sentiment).strip(),
            )

            if quad_key in seen_text_quads:
                removed_duplicates += 1
                continue

            seen_text_quads.add(quad_key)
            text_quad_counts[text_str] += 1
            kept_rows += 1
            writer.writerow(row)

    unique_texts = len(text_quad_counts)
    multi_quad_texts = sum(1 for c in text_quad_counts.values() if c > 1)
    single_quad_texts = sum(1 for c in text_quad_counts.values() if c == 1)

    print("\n" + "=" * 50)
    print("CLEANING COMPLETE")
    print("=" * 50)
    print(f"Total input rows:            {total_rows:,}")
    print(f"Removed (non-Amharic lang):  {removed_non_amharic:,}")
    print(f"Removed (no Ethiopic text):  {removed_no_ethiopic:,}")
    print(f"Removed (duplicate quads):   {removed_duplicates:,}")
    print("-" * 50)
    print(f"Total kept rows:             {kept_rows:,}")
    print(f"Unique sentences / texts:    {unique_texts:,}")
    print(f"Texts with single quad:      {single_quad_texts:,}")
    print(f"Texts with multiple quads:   {multi_quad_texts:,}")
    print(f"Cleaned file written to:     {output_path} ({os.path.getsize(output_path):,} bytes)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/Gemini_Labeled_ACOS_Data.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/Gemini_Labeled_ACOS_Data_cleaned.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    clean_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
