"""
Data Loading and Vector Store Creation Module for Argentine Spanish Learning RAG System

This module serves as the central data processing hub for the RAG system,
responsible for loading, processing, and vectorizing content from multiple data sources.

Key Capabilities:
- Loading structured data from CSV files containing Argentine Spanish phrases
- Loading content from VentureOut Spanish blog (via JSONL format)
- Converting raw data into LlamaIndex Document objects with appropriate metadata
- Creating and managing vector indices based on configuration
- Providing a unified interface for retrieving vectorized data

The module abstracts away the complexities of data source management, allowing
the application to seamlessly work with multiple content types. It handles
validation, error management, and appropriate data transformations to ensure
consistent document structure for vector embedding.

Usage:
    from core.data_loader import load_vector_store_and_data

    # Load all configured data sources and create vector store
    vector_index = load_vector_store_and_data()

    # Use the index for querying
    query_engine = vector_index.as_query_engine()
    response = query_engine.query("What is Lunfardo?")
    print(response.response)
"""

# Standard library imports
import logging
from typing import List, Optional, Union

# Third-party library imports
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore

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
VectorStore = Union[FaissVectorStore, ChromaVectorStore]
IndexType = VectorStoreIndex

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
    """Converts DataFrame rows into LlamaIndex Document objects."""
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
        doc = Document(text=content.strip(), metadata=metadata)
        documents.append(doc)
    logger.info(f"Created {len(documents)} documents from CSV DataFrame.")
    return documents


def _create_vector_index(documents: List[Document], api_key: str) -> IndexType:
    """Creates a vector index from documents using OpenAI embeddings."""
    if not documents:
        logger.error("No documents provided to create vector index.")
        raise DataLoaderError("Cannot create vector index with empty documents list.")
    try:
        import time

        # Initialize embedding model with API key and more conservative settings
        embed_model = OpenAIEmbedding(
            api_key=api_key,
            embed_batch_size=10,  # Smaller batches for rate limits
            retry_on_throttling=True,
            model="text-embedding-3-small",  # Small embedding model
            additional_kwargs={
                "dimensions": 1536  # Default embedding dimensions
            },
        )

        # Process documents in smaller batches to avoid rate limits
        logger.info(
            f"Processing {len(documents)} documents in batches to avoid rate limits"
        )
        batch_size = 20
        all_indices = []

        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            logger.info(
                f"Processing batch {i//batch_size + 1}: documents {i} to {batch_end}"
            )

            # Create an index for this batch
            batch_documents = documents[i:batch_end]

            # Add delay between batches to avoid rate limiting
            if i > 0:
                logger.info("Waiting 10 seconds to avoid rate limits...")
                time.sleep(10)

            # Create the index for this batch
            batch_index = VectorStoreIndex.from_documents(
                documents=batch_documents, embed_model=embed_model, show_progress=True
            )

            all_indices.append(batch_index)

            logger.info(f"Completed batch {i//batch_size + 1}")

        # If we processed in batches, use the first index
        # Normally we'd want to merge them, but for now this is simpler
        if all_indices:
            logger.info(
                f"Successfully created vector index with {len(all_indices)} batches"
            )
            return all_indices[0]
        else:
            logger.error("No indices were created")
            raise DataLoaderError("Failed to create any vector indices")

    except Exception as e:
        logger.error(f"Failed to create vector index: {e}", exc_info=True)
        # Raise specific error
        raise DataLoaderError(f"Failed to create vector index: {e}") from e


def _load_phrases_data() -> List[Document]:
    """
    Load and process data from the phrases CSV file.

    Returns:
        List of Document objects created from the CSV data.

    Raises:
        DataLoaderError: If loading or processing fails.
    """
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

    logger.info(f"Added {len(phrase_documents)} phrase documents to the collection.")
    return phrase_documents


def _load_ventureout_data() -> List[Document]:
    """
    Load and process VentureOut data.

    Returns:
        List of Document objects created from VentureOut data.

    Raises:
        DataLoaderError: If loading or processing fails.
    """
    # Try to copy data from data_scripts to data directory if needed
    copy_ventureout_data_to_data_dir()

    logger.info(f"Loading VentureOut data from {settings.VENTUREOUT_DATA_PATH}...")
    try:
        ventureout_data = load_ventureout_data(settings.VENTUREOUT_DATA_PATH)
        ventureout_documents = create_ventureout_documents(ventureout_data)

        if ventureout_documents:
            logger.info(
                f"Added {len(ventureout_documents)} VentureOut docs to the collection."
            )
            return ventureout_documents
        else:
            logger.warning(
                "No VentureOut documents were created. " "Continuing with phrases only."
            )
            return []
    except DataLoaderError as e:
        # Log but continue with phrases only
        logger.warning(
            f"Failed to load VentureOut data: {e}. " f"Continuing with phrases only."
        )
        return []


def _apply_debug_limit(
    documents: List[Document], debug_limit: Optional[int]
) -> List[Document]:
    """
    Apply debug limit to the number of documents if specified.

    Args:
        documents: List of Document objects to limit.
        debug_limit: Optional limit on the number of documents.

    Returns:
        List of Document objects, limited to debug_limit if specified.
    """
    if debug_limit is not None and len(documents) > debug_limit:
        logger.info(f"Limiting documents to {debug_limit} for debugging")
        return documents[:debug_limit]
    return documents


def load_vector_store_and_data(debug_limit: int = None) -> IndexType:
    """
    Loads data from configured sources, creates Documents, builds a vector index,
    and returns the index.

    Args:
        debug_limit: Optional limit on the number of documents to load for debugging

    Data sources include:
    1. Phrases CSV - Always loaded
    2. VentureOut JSONL - Loaded if USE_VENTUREOUT_DATA is True

    Returns:
        The vector index based on the loaded documents.

    Raises:
        DataLoaderError: If any step in the loading or processing fails.
    """
    logger.info("Starting data loading process...")

    try:
        # 1. Load phrase documents (required)
        all_documents = _load_phrases_data()

        # 2. Load VentureOut data if enabled and we have space for it
        has_debug_space = len(all_documents) < debug_limit if debug_limit else True
        if settings.USE_VENTUREOUT_DATA and has_debug_space:
            ventureout_docs = _load_ventureout_data()

            # Only add VentureOut docs up to the debug limit
            if debug_limit is not None and ventureout_docs:
                remaining = max(0, debug_limit - len(all_documents))
                if remaining < len(ventureout_docs):
                    logger.info(
                        f"Limiting VentureOut docs to {remaining} (debug_limit)"
                    )
                    ventureout_docs = ventureout_docs[:remaining]

            all_documents.extend(ventureout_docs)

        # 3. Apply debug limit to all documents
        all_documents = _apply_debug_limit(all_documents, debug_limit)

        # 4. Create vector index from all documents
        logger.info(
            f"Creating vector index with {len(all_documents)} total documents..."
        )
        vector_index = _create_vector_index(all_documents, settings.OPENAI_API_KEY)

        logger.info("Data loading and vector index creation complete.")
        return vector_index

    except DataLoaderError:  # Re-raise specific errors
        raise
    except Exception as e:  # Catch any other unexpected errors
        logger.exception("An unexpected error occurred during data loading.")
        raise DataLoaderError(
            "An unexpected error occurred during data loading."
        ) from e
