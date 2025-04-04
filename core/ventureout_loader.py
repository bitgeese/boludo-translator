"""
VentureOut Data Loading Module for Argentine Spanish Learning RAG System

This specialized module handles the loading, processing, and transformation
of web-scraped content from the VentureOut Spanish blog into structured
document format suitable for vector embedding.

Key Capabilities:
- Loading raw scraped data from JSONL files
- Applying advanced text cleaning to remove boilerplate content
- Processing metadata for improved retrieval relevance
- Converting cleaned data into LangChain Document objects
- Managing data file synchronization between directories

The module serves as a data adapter between the raw scraped content and
the structured format required by the vector store, ensuring that content
is properly cleaned and enriched with metadata before being made available
for retrieval.

Usage:
    from core.ventureout_loader import load_ventureout_data, create_ventureout_documents

    # Load and clean raw data
    data = load_ventureout_data("path/to/ventureout_data.jsonl")

    # Convert to Document objects
    documents = create_ventureout_documents(data)
"""

# Standard library imports
import json
import logging
import os
from typing import Any, Dict, List

# Third-party library imports
# Update to use the new LangChain import
from langchain_core.documents import Document

# Local application imports
# Import custom exception
from .exceptions import DataLoaderError

# Import text utilities
from .text_utils import clean_text

logger = logging.getLogger(__name__)


def load_ventureout_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file and apply cleaning.

    Args:
        file_path: Path to the JSONL file with VentureOut data

    Returns:
        List of dictionaries containing cleaned data

    Raises:
        DataLoaderError: If the file cannot be found or data cannot be loaded
    """
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
                        cleaned_text = clean_text(doc["text"], min_content_length=100)

                        # Skip documents with minimal content
                        # (function returns placeholder for short content)
                        if cleaned_text == "No usable content found.":
                            logger.debug(
                                f"Skipping document with minimal content: {doc['url']}"
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
    """
    Convert VentureOut data into LangChain Document objects.

    Args:
        data: List of dictionaries containing VentureOut data with cleaned text

    Returns:
        List of LangChain Document objects
    """
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

    Returns:
        bool: True if copy was successful or not needed, False if copy failed
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
