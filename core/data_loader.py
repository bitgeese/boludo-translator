"""
Data Loading and Vector Store Creation Module for Argentine Spanish Learning RAG System

This module serves as the central data processing hub for the RAG system,
responsible for loading, processing, and vectorizing content from multiple data sources.

Key Capabilities:
- Loading structured data from CSV files containing Argentine Spanish phrases
- Loading content from VentureOut Spanish blog (via JSONL format)
- Converting raw data into LangChain Document objects with appropriate metadata
- Creating and managing vector stores (FAISS or Chroma) based on configuration
- Providing a unified interface for retrieving vectorized data

The module abstracts away the complexities of data source management, allowing
the application to seamlessly work with multiple content types. It handles
validation, error management, and appropriate data transformations to ensure
consistent document structure for vector embedding.

Usage:
    from core.data_loader import load_vector_store_and_data

    # Load all configured data sources and create vector store
    vector_store = load_vector_store_and_data()

    # Perform similarity search
    results = vector_store.similarity_search("What is Lunfardo?", k=3)

    # Access retrieved documents
    for doc in results:
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content)
"""

# Standard library imports
import logging
import os
from typing import List, Union

# Third-party library imports
import pandas as pd
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Local application imports
# Import the settings object
from config import settings

# Import custom exception
from .exceptions import DataLoaderError

# Import VentureOut data loader
from .ventureout_loader import (
    copy_ventureout_data_to_data_dir,
    create_ventureout_documents,
    load_ventureout_data,
)

# Type alias for vector stores
VectorStore = Union[FAISS, Chroma]

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
            "source": "phrases_csv",  # Add source to identify origin
            "data_type": "phrase",  # Add type for potential filtering
        }
        # Ensure all metadata values are strings for vector store compatibility
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)
    logger.info(f"Created {len(documents)} documents from CSV DataFrame.")
    return documents


def _create_vector_store(documents: List[Document], api_key: str) -> VectorStore:
    """Creates a vector store from documents using OpenAI embeddings."""
    if not documents:
        logger.error("No documents provided to create vector store.")
        raise DataLoaderError("Cannot create vector store with empty documents list.")
    try:
        # Ensure api_key is passed explicitly for clarity
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

        # Choose vector store type based on settings
        if settings.VECTORSTORE_TYPE.upper() == "CHROMA":
            # Ensure directory exists
            os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            )
            # Persist to disk
            vector_store.persist()
            logger.info(
                f"Successfully created Chroma vector store in "
                f"{settings.CHROMA_PERSIST_DIRECTORY}."
            )
        else:
            # Default to FAISS
            vector_store = FAISS.from_documents(documents, embedding_model)
            logger.info("Successfully created FAISS vector store.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}", exc_info=True)
        # Raise specific error
        raise DataLoaderError(f"Failed to create vector store: {e}") from e


def load_vector_store_and_data() -> VectorStore:
    """
    Loads data from configured sources, creates Documents, builds a vector store,
    and returns the store.

    Data sources include:
    1. Phrases CSV - Always loaded
    2. VentureOut JSONL - Loaded if USE_VENTUREOUT_DATA is True

    Returns:
        The vector store (FAISS or Chroma based on settings).

    Raises:
        DataLoaderError: If any step in the loading or processing fails.
    """
    logger.info("Starting data loading process...")
    all_documents = []

    try:
        # 1. Load and process phrases CSV (always required)
        logger.info(f"Loading phrases from {settings.PHRASES_CSV_PATH}...")
        df = _load_data_from_csv(settings.PHRASES_CSV_PATH)
        if df.empty:
            raise DataLoaderError(
                f"No data loaded from {settings.PHRASES_CSV_PATH}. Cannot proceed."
            )

        phrase_documents = _create_documents_from_dataframe(df)
        if not phrase_documents:
            raise DataLoaderError(
                "No documents were created from the CSV DataFrame. Cannot proceed."
            )

        all_documents.extend(phrase_documents)
        logger.info(
            f"Added {len(phrase_documents)} phrase documents to the collection."
        )

        # 2. Load and process VentureOut data if enabled
        if settings.USE_VENTUREOUT_DATA:
            # Try to copy data from data_scripts to data directory if needed
            copy_ventureout_data_to_data_dir()

            logger.info(
                f"Loading VentureOut data from {settings.VENTUREOUT_DATA_PATH}..."
            )
            try:
                ventureout_data = load_ventureout_data(settings.VENTUREOUT_DATA_PATH)
                ventureout_documents = create_ventureout_documents(ventureout_data)

                if ventureout_documents:
                    all_documents.extend(ventureout_documents)
                    logger.info(
                        f"Added {len(ventureout_documents)} VentureOut documents "
                        f"to the collection."
                    )
                else:
                    logger.warning(
                        "No VentureOut documents were created. "
                        "Continuing with phrases only."
                    )
            except DataLoaderError as e:
                # Log but continue with phrases only
                logger.warning(
                    f"Failed to load VentureOut data: {e}. "
                    f"Continuing with phrases only."
                )

        # 3. Create vector store from all documents
        logger.info(
            f"Creating vector store with {len(all_documents)} total documents..."
        )
        vector_store = _create_vector_store(all_documents, settings.OPENAI_API_KEY)

        logger.info("Data loading and vector store creation complete.")
        # Return the vector store
        return vector_store
    except DataLoaderError:  # Re-raise specific errors
        raise
    except Exception as e:  # Catch any other unexpected errors
        logger.exception("An unexpected error occurred during data loading.")
        raise DataLoaderError(
            "An unexpected error occurred during data loading."
        ) from e
