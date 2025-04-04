"""
VentureOut Data Loading Module

This module provides functions to load VentureOut Spanish data from a JSONL file,
clean the content, and process it into LangChain Documents.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain.docstore.document import Document

# Import custom exception
from .exceptions import DataLoaderError

logger = logging.getLogger(__name__)

# Common patterns to remove from the text content for cleaning
PATTERNS_TO_REMOVE = [
    # Comment form and related elements
    r"Leave a Reply\s*Cancel reply.*?for the next time I comment\.",
    r"Comment\s*Enter your name.*?for the next time I comment\.",
    r"Enter your name or username to comment.*?for the next time I comment\.",
    # WordPress footer elements and metadata
    r"Copyright © \d{4}.*?All rights reserved\.?",
    r"Post author:\s*.*?\s*Post published:\s*.*?\s*Reading time:\s*.*?\s*",
    r"Thank you for sharing this post!\s*Share this content\s*Opens in a new window\s*",
    # Navigation elements
    r"Previous Post.*?Next Post",
    # Social sharing
    r"Share this:.*?Click to share",
    r"Share this content.*?Opens in a new window",
    # Search prompts
    r"Search for:.*?search",
    # Category and tag listings at the end of posts
    r"Categories.*?\(\d+\).*?Spanish Teaching.*?\(\d+\)",
    r"\(\d+\)\s*Argentinian Spanish\s*\(\d+\)\s*Argentinian Spanish Curse Words"
    r"\s*\(\d+\)",
    # Common footer text
    r"Venture Out Spanish\s*·\s*\S+\s*\S+\s*·\s*Privacy Policy",
    # Author bio section
    r"About the author.*?View all posts",
    # Comment section headers
    r"Comments\s*\(\d+\)",
    # WordPress tags
    r"Filed under:.*?Tags:",
    # Post metadata block
    r"Post author:.*?Reading time:.*?read",
    # Share buttons
    r"Opens in a new window\s*Opens in a new window\s*Opens in a new window",
]


def clean_ventureout_text(text: str) -> str:
    """Clean up text content by removing common boilerplate elements."""
    # First, remove any identified patterns
    for pattern in PATTERNS_TO_REMOVE:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Split text at common ending points
    split_points = [
        "Leave a Reply",
        "Related Posts",
        "Categories",
        "Thank you for sharing this post",
        "Post navigation",
    ]

    for point in split_points:
        if point in text:
            text = text.split(point)[0]

    # Remove category listings (common at the end of posts)
    lines = text.splitlines()
    filtered_lines = []
    skip_mode = False
    category_pattern = re.compile(r"^\s*\(\d+\)\s*$")

    for line in lines:
        # If line is like "(45)" - part of category listings - enter skip mode
        if category_pattern.match(line):
            skip_mode = True

        # If we're not in skip mode, keep the line
        if not skip_mode:
            filtered_lines.append(line)

        # If we encounter a long line after categories, exit skip mode
        if skip_mode and len(line.strip()) > 30:
            skip_mode = False

    # Rejoin the filtered lines
    text = "\n".join(filtered_lines)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Remove any URLs that might be in the text
    text = re.sub(r"https?://\S+", "", text)

    return text.strip()


def load_ventureout_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file and apply cleaning."""
    logger.info(f"Loading and cleaning VentureOut data from {file_path}")
    documents = []
    processed_count = 0
    skipped_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        cleaned_text = clean_ventureout_text(doc["text"])

                        # Skip documents with minimal content
                        if len(cleaned_text) < 100:
                            logger.debug(
                                f"Skipping document with minimal content: "
                                f"{doc['url']}"
                            )
                            skipped_count += 1
                            continue

                        # Add cleaned text and source info
                        doc["cleaned_text"] = cleaned_text
                        documents.append(doc)
                        processed_count += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON: {e}")
                        skipped_count += 1
                    except KeyError as e:
                        logger.error(f"Missing key {e} in JSONL line: {line[:100]}")
                        skipped_count += 1

        logger.info(
            f"Loaded and cleaned {processed_count} documents from {file_path}. "
            f"Skipped {skipped_count}."
        )
        return documents
    except FileNotFoundError:
        logger.error(f"VentureOut data file not found at {file_path}")
        raise DataLoaderError(f"VentureOut data file not found at {file_path}")
    except Exception as e:
        logger.error(
            f"Error loading VentureOut data from {file_path}: {e}", exc_info=True
        )
        raise DataLoaderError(f"Failed to load VentureOut data: {e}")


def create_ventureout_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert VentureOut data into LangChain Document objects."""
    documents = []
    for item in data:
        # Create meaningful content with title and text
        content = f"""
        Title: {item.get("title", "Untitled")}
        Source: VentureOut Spanish
        URL: {item.get("url", "")}
        Content: {item.get("cleaned_text", "")}
        """

        # Metadata for retrieval and filtering
        metadata = {
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "source": "VentureOut Spanish",
            "data_type": "article",
            "region": "Argentina",  # Assuming all content is related to Argentina
            "formality": "Various",  # Could be refined if we had specific info
        }

        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)

    logger.info(f"Created {len(documents)} VentureOut documents.")
    return documents


def copy_ventureout_data_to_data_dir():
    """
    Copy ventureout_data.jsonl from data_scripts to data directory if it exists.
    This ensures the data is in the expected location for the app to use.
    """
    source_path = "data_scripts/ventureout_data.jsonl"
    target_path = "data/ventureout_data.jsonl"

    # Check if the source file exists
    if not os.path.exists(source_path):
        logger.warning(f"Source file {source_path} not found. Skipping copy operation.")
        return False

    # Ensure the target directory exists
    os.makedirs("data", exist_ok=True)

    try:
        # If target already exists, check if we need to update it
        if os.path.exists(target_path):
            source_mtime = os.path.getmtime(source_path)
            target_mtime = os.path.getmtime(target_path)

            if source_mtime <= target_mtime:
                logger.info(f"Target file {target_path} is up to date. Skipping copy.")
                return True

        # Copy the file
        import shutil

        shutil.copy2(source_path, target_path)
        logger.info(f"Successfully copied {source_path} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying VentureOut data: {e}", exc_info=True)
        return False
