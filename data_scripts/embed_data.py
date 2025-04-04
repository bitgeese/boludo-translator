#!/usr/bin/env python3
"""
Processing script for ventureout_data.jsonl AND CSV data to create embeddings and store
them in a unified vector database.

This script:
1. Reads scraped data from the JSONL file
2. Reads data from a specified CSV file
3. Cleans the text content from JSONL to remove boilerplate
4. Combines data from both sources
5. Splits combined text into chunks using Langchain's text splitters
6. Creates embeddings for each chunk
7. Stores the embeddings in a Chroma vector database with metadata indicating the source
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import pandas as pd
from langchain.docstore.document import Document

# Langchain imports - updated to fix deprecation warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
JSONL_INPUT_FILE = "ventureout_data.jsonl"
CSV_INPUT_FILE = "argentine_spanish_qa.csv"  # Assumed CSV filename
CHROMA_PERSIST_DIRECTORY = "../chroma_db"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# --- CSV Configuration (Adjust column names as needed) ---
CSV_TEXT_COLUMN = "Answer"  # Column containing the main text/answer
CSV_METADATA_COLUMNS = [
    "Question",
    "Source",
]  # Columns to use as metadata (if they exist)

# Common patterns to remove from the JSONL text
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
    (
        r"\(\d+\)\s*Argentinian Spanish\s*\(\d+\)\s*Argentinian Spanish Curse "
        r"Words\s*\(\d+\)"
    ),
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


def clean_jsonl_text(text: str) -> str:
    """Clean up text content from JSONL source by removing common boilerplate
    elements.
    """
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

    # Handle missing content
    if not text or len(text.strip()) < 10:  # Arbitrary threshold
        return "No usable content found."

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Remove any URLs that might be in the text
    text = re.sub(r"https?://\S+", "", text)

    return text.strip()


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file and apply cleaning."""
    logging.info(f"Loading and cleaning data from {file_path}")
    documents = []
    processed_count = 0
    skipped_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    cleaned_text = clean_jsonl_text(doc["text"])

                    if len(cleaned_text) < 100:  # Skip docs with minimal content
                        logging.warning(
                            f"Skipping JSONL document with minimal content: "
                            f"{doc['url']}"
                        )
                        skipped_count += 1
                        continue

                    # Add cleaned text and identify source
                    doc["cleaned_text"] = cleaned_text
                    doc["origin"] = "ventureoutspanish.com"  # Add origin field
                    documents.append(doc)
                    processed_count += 1
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON: {e}")
                except KeyError as e:
                    logging.error(f"Missing key {e} in JSONL line: {line.strip()}")
                    skipped_count += 1

    logging.info(
        f"Loaded and cleaned {processed_count} documents from {file_path}. "
        f"Skipped {skipped_count}."
    )
    return documents


def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    logging.info(f"Loading data from {file_path}")
    documents = []
    processed_count = 0
    skipped_count = 0
    try:
        df = pd.read_csv(file_path)
        # Validate required column exists
        if CSV_TEXT_COLUMN not in df.columns:
            logging.error(
                f"Required column '{CSV_TEXT_COLUMN}' not found in {file_path}. "
                f"Skipping CSV."
            )
            return []

        for index, row in df.iterrows():
            text_content = row[CSV_TEXT_COLUMN]
            if (
                not isinstance(text_content, str) or len(text_content.strip()) < 20
            ):  # Skip short/invalid text
                logging.warning(
                    f"Skipping CSV row {index + 2} due to short/invalid content in "
                    f"'{CSV_TEXT_COLUMN}'."
                )
                skipped_count += 1
                continue

            # Prepare metadata, checking if columns exist
            metadata = {
                col: row[col]
                for col in CSV_METADATA_COLUMNS
                if col in df.columns and pd.notna(row[col])
            }
            metadata["origin"] = "CSV"  # Add origin field

            doc = {
                "cleaned_text": text_content.strip(),  # CSV text is clean
                **metadata,  # Unpack metadata dict here
            }
            documents.append(doc)
            processed_count += 1

    except FileNotFoundError:
        logging.warning(f"CSV file not found at {file_path}. Skipping CSV loading.")
        return []
    except Exception as e:
        logging.error(f"Error reading CSV file {file_path}: {e}")
        return []

    logging.info(
        f"Loaded {processed_count} documents from {file_path}. "
        f"Skipped {skipped_count}."
    )
    return documents


def create_langchain_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert combined data into Langchain Document objects."""
    langchain_docs = []
    required_keys = ["cleaned_text", "origin"]  # Minimal required keys

    for item in data:
        # Basic validation
        if not all(key in item for key in required_keys) or not item["cleaned_text"]:
            logging.warning(
                f"Skipping item due to missing required keys or empty text: "
                f"{item.get('url', item.get('Question', 'Unknown Item'))}"
            )
            continue

        # Extract page content and prepare metadata
        page_content = item["cleaned_text"]
        metadata = {k: v for k, v in item.items() if k != "cleaned_text"}

        # Create Document
        doc = Document(page_content=page_content, metadata=metadata)
        langchain_docs.append(doc)

    logging.info(f"Created {len(langchain_docs)} Langchain documents for embedding")
    return langchain_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    if not documents:
        logging.warning("No documents to split.")
        return []
    logging.info(f"Splitting {len(documents)} documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} chunks")
    return split_docs


def create_vector_store(documents: List[Document]) -> Chroma:
    """Create and populate a Chroma vector store with documents."""
    if not documents:
        logging.error("No documents provided to create vector store.")
        return None

    logging.info(f"Creating vector store with {len(documents)} chunks...")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()  # Requires OPENAI_API_KEY env variable

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    vectorstore.persist()

    logging.info(f"Vector store created and persisted to {CHROMA_PERSIST_DIRECTORY}")
    return vectorstore


def main():
    """Main execution function."""
    # Check if OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        logging.error(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before running this script."
        )
        return

    # Load data from JSONL
    jsonl_data = load_jsonl_data(JSONL_INPUT_FILE)

    # Load data from CSV
    csv_data = load_csv_data(CSV_INPUT_FILE)

    # Combine data
    combined_data = jsonl_data + csv_data
    logging.info(f"Total documents combined from both sources: {len(combined_data)}")

    if not combined_data:
        logging.error("No data loaded from any source. Exiting.")
        return

    # Convert to Langchain documents
    langchain_docs = create_langchain_documents(combined_data)

    # Split into chunks
    split_docs = split_documents(langchain_docs)

    # Create vector store
    vector_store = create_vector_store(split_docs)

    if vector_store:
        logging.info("Processing complete!")
        # Example of how to query the vector store
        logging.info("Example query: 'What is Lunfardo?'")
        try:
            results = vector_store.similarity_search("What is Lunfardo?", k=2)
            for i, doc in enumerate(results):
                logging.info(f"--- Result {i + 1} ---")
                logging.info(f"Origin: {doc.metadata.get('origin')}")
                logging.info(
                    f"Source: {doc.metadata.get('url') or doc.metadata.get('Source')}"
                )
                logging.info(
                    f"Title/Question: "
                    f"{doc.metadata.get('title') or doc.metadata.get('Question')}"
                )
                logging.info(f"Content snippet: {doc.page_content[:150]}...")
        except Exception as e:
            logging.error(f"Error running example query: {e}")
    else:
        logging.error("Failed to create vector store.")


if __name__ == "__main__":
    main()
