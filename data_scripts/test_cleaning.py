#!/usr/bin/env python3
"""
Test script to verify the text cleaning function.
This loads a few samples from ventureout_data.jsonl and shows the before/after cleaning.
"""

import json
import os
import sys

# Add parent directory to path to be able to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.text_utils import clean_text
except ImportError:
    # If import fails, show a helpful error message
    print(
        "Failed to import text_utils from core. This script should be run from "
        "the project root directory or with the project root in PYTHONPATH."
    )

    # Define a fallback clean_text function if import fails
    def clean_text(text, min_content_length=10):
        """Fallback function if text_utils cannot be imported"""
        return text.strip()


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
            cleaned_text = clean_text(original_text, min_content_length=100)

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
