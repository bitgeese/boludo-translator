"""
Data Loading and Vector Store Creation Module

This module provides functions to load Argentinian Spanish phrases from a CSV,
process them into LangChain Documents, and create a FAISS vector store.
"""

import logging
from typing import List

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Import the settings object
from config import settings

# Import custom exception
from .exceptions import DataLoaderError

logger = logging.getLogger(__name__)


def _load_data_from_csv(file_path: str) -> pd.DataFrame:
    """Loads data from the specified CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} phrases from {file_path}")
        # Ensure required columns exist
        required_cols = [
            "Original Phrase/Word",
            "Argentinian Equivalent",
            "Explanation (Context/Usage)",
            "Region Specificity",
            "Level of Formality",
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            # Raise specific error
            raise DataLoaderError(f"CSV missing required columns: {missing}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {file_path}")
        # Raise specific error
        raise DataLoaderError(f"CSV file not found at {file_path}") from None
    except ValueError as e:
        # Catch potential value errors from validation and wrap them
        logger.error(f"Data validation error in CSV {file_path}: {e}")
        raise DataLoaderError(f"Data validation error in CSV: {e}") from e
    except Exception as e:
        logger.error(
            f"Error loading or validating CSV from {file_path}: {e}", exc_info=True
        )
        # Raise specific error, chaining the original exception
        raise DataLoaderError(f"Failed to load or validate CSV: {e}") from e


def _create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    """Converts DataFrame rows into LangChain Document objects."""
    documents = []
    # Define expected columns for clarity
    expected_cols = [
        "Original Phrase/Word",
        "Argentinian Equivalent",
        "Explanation (Context/Usage)",
        "Region Specificity",
        "Level of Formality",
        "Example Sentence (Spanish)",
        "Example Sentence (English)",
        "Connotation",
        "Register",
    ]
    # Check if expected columns exist in the DataFrame after enrichment
    if not all(col in df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in df.columns]
        logger.error(
            f"Enriched DataFrame is missing expected columns: {missing}. "
            f"Cannot create documents."
        )
        # Raise specific error
        raise DataLoaderError(
            f"Enriched DataFrame is missing expected columns: {missing}"
        )

    for index, row in df.iterrows():
        # Include enriched fields in the page_content
        # Use .get() with defaults for robustness against missing data
        # (though checked above)
        content = f"""
        Original: {row.get("Original Phrase/Word", "")}
        Argentinian: {row.get("Argentinian Equivalent", "")}
        Context/Explanation: {row.get("Explanation (Context/Usage)", "")}
        Region: {row.get("Region Specificity", "Unknown")}
        Register: {row.get("Register", "Unknown")}
        Connotation: {row.get("Connotation", "Unknown")}
        Example (Spanish): {row.get("Example Sentence (Spanish)", "")}
        Example (English): {row.get("Example Sentence (English)", "")}
        Formality: {row.get("Level of Formality", "Unknown")} # Keep original?
        """
        metadata = {
            "original": str(row.get("Original Phrase/Word", "")),
            "argentinian": str(row.get("Argentinian Equivalent", "")),
            "context": str(row.get("Explanation (Context/Usage)", "")),
            "region": str(row.get("Region Specificity", "Unknown")),
            "formality": str(row.get("Level of Formality", "Unknown")),
            # Add new fields to metadata as well for potential filtering later
            "register": str(row.get("Register", "Unknown")),
            "connotation": str(row.get("Connotation", "Unknown")),
        }
        # Ensure all metadata values are strings for FAISS compatibility
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)
    logger.info(f"Created {len(documents)} documents from DataFrame.")
    return documents


def _create_vector_store(documents: List[Document], api_key: str) -> FAISS:
    """Creates a FAISS vector store from documents using OpenAI embeddings."""
    try:
        # Ensure api_key is passed explicitly for clarity
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(documents, embedding_model)
        logger.info("Successfully created FAISS vector store.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}", exc_info=True)
        # Raise specific error
        raise DataLoaderError(f"Failed to create FAISS vector store: {e}") from e


def load_vector_store_and_data() -> FAISS:
    """
    Loads data from the configured CSV, creates Documents, builds a FAISS
    vector store, and returns the store.

    Uses configuration from the `config.settings` object.

    Returns:
        The FAISS vector store.

    Raises:
        DataLoaderError: If any step in the loading or processing fails.
    """
    logger.info(f"Starting data loading process from {settings.PHRASES_CSV_PATH}...")
    try:
        df = _load_data_from_csv(settings.PHRASES_CSV_PATH)
        if df.empty:
            raise DataLoaderError(
                f"No data loaded from {settings.PHRASES_CSV_PATH}. Cannot proceed."
            )

        documents = _create_documents_from_dataframe(df)
        if not documents:
            raise DataLoaderError(
                "No documents were created from the DataFrame. Cannot proceed."
            )

        vector_store = _create_vector_store(documents, settings.OPENAI_API_KEY)

        logger.info("Data loading and vector store creation complete.")
        # Return only the vector store
        return vector_store
    except DataLoaderError:  # Re-raise specific errors
        raise
    except Exception as e:  # Catch any other unexpected errors
        logger.exception("An unexpected error occurred during data loading.")
        raise DataLoaderError(
            "An unexpected error occurred during data loading."
        ) from e
