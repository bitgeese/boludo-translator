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

# Standard library imports
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Third-party library imports
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local application imports
# Add parent directory to path to be able to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.text_utils import clean_text
    from data_scripts.config import (
        CHROMA_PERSIST_DIRECTORY,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        CSV_INPUT_FILE,
        CSV_METADATA_COLUMNS,
        CSV_TEXT_COLUMN,
        JSONL_INPUT_FILE,
        MIN_CONTENT_LENGTH,
        log_config,
    )
    from data_scripts.exceptions import (
        ConfigurationError,
        DataLoadingError,
        DataProcessingError,
        EmbeddingError,
        VectorStoreError,
    )
except ImportError as e:
    # If import fails, show a helpful error message
    logging.error(
        f"Failed to import required modules: {e}. This script should be run from "
        "the project root directory or with the project root in PYTHONPATH."
    )

    # Define fallback exception classes if import fails
    class DataLoadingError(Exception):
        pass

    class DataProcessingError(Exception):
        pass

    class EmbeddingError(Exception):
        pass

    class VectorStoreError(Exception):
        pass

    class ConfigurationError(Exception):
        pass

    # Define a fallback clean_text function if import fails
    def clean_text(text: str, min_content_length: int = 10) -> str:
        """Fallback function if text_utils cannot be imported"""
        return text.strip()

    # Define fallback configuration constants
    JSONL_INPUT_FILE = "ventureout_data.jsonl"
    CSV_INPUT_FILE = "argentine_spanish_qa.csv"
    CHROMA_PERSIST_DIRECTORY = "../chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MIN_CONTENT_LENGTH = 100
    CSV_TEXT_COLUMN = "Answer"
    CSV_METADATA_COLUMNS = ["Question", "Source"]

    def log_config():
        pass


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file and apply cleaning.

    Args:
        file_path: Path to the JSONL file to load

    Returns:
        List of dictionaries containing document data

    Raises:
        DataLoadingError: If there's an issue loading or parsing the JSONL file
    """
    logging.info(f"Loading and cleaning data from {file_path}")
    documents = []
    processed_count = 0
    skipped_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        cleaned_text = clean_text(doc["text"], min_content_length=100)

                        if (
                            cleaned_text == "No usable content found."
                        ):  # Skip docs with minimal content
                            logging.warning(
                                f"Skipping JSONL document with minimal content: "
                                f"{doc.get('url', f'Line {line_num}')}"
                            )
                            skipped_count += 1
                            continue

                        # Add cleaned text and identify source
                        doc["cleaned_text"] = cleaned_text
                        doc["origin"] = "ventureoutspanish.com"  # Add origin field
                        documents.append(doc)
                        processed_count += 1
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing JSON at line {line_num}: {e}"
                        logging.error(error_msg)
                        # Continue processing other lines, don't raise exception here
                    except KeyError as e:
                        error_msg = (
                            f"Missing key {e} in JSONL at line {line_num}: "
                            f"{line[:50]}..."
                        )
                        logging.error(error_msg)
                        skipped_count += 1
                        # Continue processing other lines, don't raise exception here

        if processed_count == 0 and skipped_count > 0:
            # If all lines were skipped, raise an exception
            raise DataLoadingError(
                f"Failed to load any valid documents from {file_path}. "
                f"All {skipped_count} entries were skipped due to errors."
            )

        logging.info(
            f"Loaded and cleaned {processed_count} documents from {file_path}. "
            f"Skipped {skipped_count}."
        )
        return documents

    except FileNotFoundError as e:
        error_msg = f"JSONL file not found at {file_path}"
        logging.error(error_msg)
        raise DataLoadingError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error loading JSONL from {file_path}: {e}"
        logging.error(error_msg, exc_info=True)
        raise DataLoadingError(error_msg) from e


def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.

    Args:
        file_path: Path to the CSV file to load

    Returns:
        List of dictionaries containing document data

    Raises:
        DataLoadingError: If there's an issue loading or processing the CSV file
    """
    logging.info(f"Loading data from {file_path}")
    documents = []
    processed_count = 0
    skipped_count = 0

    try:
        df = pd.read_csv(file_path)
        # Validate required column exists
        if CSV_TEXT_COLUMN not in df.columns:
            error_msg = (
                f"Required column '{CSV_TEXT_COLUMN}' not found in {file_path}. "
                f"Available columns: {', '.join(df.columns)}"
            )
            logging.error(error_msg)
            raise DataLoadingError(error_msg)

        for index, row in df.iterrows():
            text_content = row[CSV_TEXT_COLUMN]
            if not isinstance(text_content, str) or len(text_content.strip()) < 20:
                # Skip short/invalid text
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

        if processed_count == 0 and skipped_count > 0:
            # If all rows were skipped, raise an exception
            raise DataLoadingError(
                f"Failed to load any valid documents from {file_path}. "
                f"All {skipped_count} entries were skipped due to validation issues."
            )

        logging.info(
            f"Loaded {processed_count} documents from {file_path}. "
            f"Skipped {skipped_count}."
        )
        return documents

    except FileNotFoundError:
        msg = f"CSV file not found at {file_path}"
        logging.warning(msg)
        # Return empty list instead of raising, as CSV is optional
        return []
    except DataLoadingError:
        # Re-raise DataLoadingError exceptions
        raise
    except Exception as e:
        error_msg = f"Error reading CSV file {file_path}: {e}"
        logging.error(error_msg, exc_info=True)
        raise DataLoadingError(error_msg) from e


def create_langchain_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert combined data into Langchain Document objects.

    Args:
        data: List of dictionaries containing document data with cleaned text

    Returns:
        List of Langchain Document objects

    Raises:
        DataProcessingError: If there's an error processing the data
    """
    langchain_docs: List[Document] = []
    required_keys: List[str] = ["cleaned_text", "origin"]  # Minimal required keys
    skipped_count: int = 0

    try:
        for item in data:
            # Basic validation
            if (
                not all(key in item for key in required_keys)
                or not item["cleaned_text"]
            ):
                logging.warning(
                    f"Skipping item due to missing required keys or empty text: "
                    f"{item.get('url', item.get('Question', 'Unknown Item'))}"
                )
                skipped_count += 1
                continue

            # Extract page content and prepare metadata
            page_content: str = item["cleaned_text"]
            metadata: Dict[str, Any] = {
                k: v for k, v in item.items() if k != "cleaned_text"
            }

            # Create Document
            doc = Document(page_content=page_content, metadata=metadata)
            langchain_docs.append(doc)

        if not langchain_docs and len(data) > 0:
            # We had data but couldn't create any documents
            raise DataProcessingError(
                f"Failed to create any valid documents from {len(data)} data items. "
                f"Skipped {skipped_count} items due to missing required keys or "
                f"empty text."
            )

        logging.info(
            f"Created {len(langchain_docs)} Langchain documents for embedding. "
            f"Skipped {skipped_count} items."
        )
        return langchain_docs
    except Exception as e:
        if isinstance(e, DataProcessingError):
            # Re-raise DataProcessingError exceptions
            raise
        error_msg = f"Error creating Langchain documents: {e}"
        logging.error(error_msg, exc_info=True)
        raise DataProcessingError(error_msg) from e


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of Document objects to split

    Returns:
        List of split Document chunks

    Raises:
        DataProcessingError: If there's an error splitting the documents
    """
    if not documents:
        logging.warning("No documents to split.")
        return []

    try:
        logging.info(f"Splitting {len(documents)} documents into chunks")
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs: List[Document] = text_splitter.split_documents(documents)

        if not split_docs:
            # This is unlikely but could happen with very short documents
            raise DataProcessingError(
                f"Failed to create any chunks from {len(documents)} documents. "
                f"Check your chunk size settings or document content."
            )

        logging.info(f"Created {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        if isinstance(e, DataProcessingError):
            # Re-raise DataProcessingError exceptions
            raise
        error_msg = f"Error splitting documents: {e}"
        logging.error(error_msg, exc_info=True)
        raise DataProcessingError(error_msg) from e


def create_vector_store(documents: List[Document]) -> Optional[Chroma]:
    """
    Create and populate a Chroma vector store with documents.

    Args:
        documents: List of Document objects to embed and store

    Returns:
        Populated Chroma vector store or None if creation fails

    Raises:
        VectorStoreError: If there's an error creating the vector store
        ConfigurationError: If required API key is missing
    """
    if not documents:
        error_msg = "No documents provided to create vector store."
        logging.error(error_msg)
        raise VectorStoreError(error_msg)

    try:
        logging.info(f"Creating vector store with {len(documents)} chunks...")

        # Verify that the OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            error_msg = (
                "OPENAI_API_KEY environment variable not set. "
                "Please set it before running this script."
            )
            logging.error(error_msg)
            raise ConfigurationError(error_msg)

        # Initialize embeddings
        embeddings: OpenAIEmbeddings = (
            OpenAIEmbeddings()
        )  # Requires OPENAI_API_KEY env variable

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(CHROMA_PERSIST_DIRECTORY), exist_ok=True)

        # Create and persist the vector store
        vectorstore: Chroma = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        vectorstore.persist()

        logging.info(
            f"Vector store created and persisted to {CHROMA_PERSIST_DIRECTORY}"
        )
        return vectorstore
    except ValueError as e:
        # This could be from bad embeddings or other validation issues
        error_msg = f"Value error creating vector store: {e}"
        logging.error(error_msg, exc_info=True)
        raise VectorStoreError(error_msg) from e
    except Exception as e:
        if isinstance(e, (VectorStoreError, ConfigurationError)):
            # Re-raise our custom exceptions
            raise
        error_msg = f"Error creating vector store: {e}"
        logging.error(error_msg, exc_info=True)
        raise VectorStoreError(error_msg) from e


def main():
    """
    Main execution function to run the data processing pipeline.

    Steps:
    1. Load data from JSONL and CSV sources
    2. Create Langchain documents
    3. Split documents into chunks
    4. Create vector store with embeddings
    """
    # Check if OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        logging.error(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before running this script."
        )
        return

    # Log configuration settings
    log_config()

    try:
        # Load data from JSONL
        jsonl_data = load_jsonl_data(JSONL_INPUT_FILE)

        # Load data from CSV (this will return an empty list if file not found)
        csv_data = load_csv_data(CSV_INPUT_FILE)

        # Combine data
        combined_data = jsonl_data + csv_data
        logging.info(
            f"Total documents combined from both sources: {len(combined_data)}"
        )

        if not combined_data:
            raise DataLoadingError("No data loaded from any source. Cannot proceed.")

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
                    url_source = doc.metadata.get("url") or doc.metadata.get("Source")
                    logging.info(f"Source: {url_source}")
                    logging.info(
                        f"Title/Question: "
                        f"{doc.metadata.get('title') or doc.metadata.get('Question')}"
                    )
                    logging.info(f"Content snippet: {doc.page_content[:150]}...")
            except Exception as e:
                logging.error(f"Error running example query: {e}")
        else:
            logging.error("Failed to create vector store.")
    except (
        DataLoadingError,
        DataProcessingError,
        VectorStoreError,
        ConfigurationError,
    ) as e:
        # Log the error and exit with a non-zero status
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other exceptions and provide helpful context
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
