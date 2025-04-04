#!/usr/bin/env python3
"""
Test script to verify the text cleaning function.
This loads a few samples from ventureout_data.jsonl and shows the before/after cleaning.
"""

import json

from embed_data import clean_text  # Import the clean_text function from embed_data.py


def main():
    """Load a few samples from the JSONL file and test the cleaning function."""
    input_file = "ventureout_data.jsonl"

    try:
        # Try to load the first 2 documents
        with open(input_file, "r", encoding="utf-8") as f:
            samples = []
            for i, line in enumerate(f):
                if i >= 2:  # Just test first 2 documents
                    break
                if line.strip():
                    samples.append(json.loads(line))

        if not samples:
            print(f"No documents found in {input_file}")
            return

        # Process each sample
        for i, sample in enumerate(samples):
            original_text = sample["text"]
            cleaned_text = clean_text(original_text)

            print(f"\n--- DOCUMENT {i + 1}: {sample['title']} ---")
            print(f"URL: {sample['url']}")

            # Print length comparison
            print(f"Original length: {len(original_text)} characters")
            print(f"Cleaned length: {len(cleaned_text)} characters")
            print(
                f"Reduction: {(1 - len(cleaned_text) / len(original_text)) * 100:.2f}%"
            )

            # Show a short sample of the start and end of both texts
            print("\nORIGINAL TEXT (first 200 chars):")
            print(original_text[:200] + "...")
            print("\nORIGINAL TEXT (last 200 chars):")
            print("..." + original_text[-200:])

            print("\nCLEANED TEXT (first 200 chars):")
            print(cleaned_text[:200] + "...")
            print("\nCLEANED TEXT (last 200 chars):")
            print("..." + cleaned_text[-200:])

            print("\n" + "=" * 80)

    except FileNotFoundError:
        print(f"File not found: {input_file}")
        print("Make sure you've run scrape_ventureout.py first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
