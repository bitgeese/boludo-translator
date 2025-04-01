"""
Data Loading and Vector Store Creation Module

This module provides functions to load Argentinian Spanish phrases from a CSV,
process them into LangChain Documents, and create a FAISS vector store.
"""

import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import logging

from config import PHRASES_CSV_PATH, OPENAI_API_KEY

logger = logging.getLogger(__name__)

def _load_data_from_csv(file_path: str) -> pd.DataFrame:
    """Loads data from the specified CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} phrases from {file_path}")
        # Ensure required columns exist
        required_cols = [
            'Original Phrase/Word', 'Argentinian Equivalent', 
            'Explanation (Context/Usage)', 'Region Specificity', 'Level of Formality'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"CSV missing required columns: {missing}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading or validating CSV from {file_path}: {e}")
        raise

def _create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    """Converts DataFrame rows into LangChain Document objects."""
    documents = []
    for _, row in df.iterrows():
        content = f"""
        Original: {row['Original Phrase/Word']}
        Argentinian: {row['Argentinian Equivalent']}
        Context: {row['Explanation (Context/Usage)']}
        Region: {row['Region Specificity']}
        Formality: {row['Level of Formality']}
        """
        metadata = {
            "original": str(row['Original Phrase/Word']),
            "argentinian": str(row['Argentinian Equivalent']),
            "context": str(row['Explanation (Context/Usage)']),
            "region": str(row['Region Specificity']),
            "formality": str(row['Level of Formality'])
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
        logger.error(f"Failed to create vector store: {e}")
        raise

def load_vector_store_and_data() -> Tuple[FAISS, pd.DataFrame]:
    """
    Loads data from the configured CSV, creates Documents, builds a FAISS
    vector store, and returns the store and the original DataFrame.

    Uses configuration from the `config` module.

    Returns:
        A tuple containing the FAISS vector store and the reference DataFrame.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If the CSV is missing required columns or data is invalid.
        Exception: For other potential errors during loading or embedding.
    """
    logger.info(f"Starting data loading process from {PHRASES_CSV_PATH}...")
    df = _load_data_from_csv(PHRASES_CSV_PATH)
    if df.empty:
        raise ValueError(f"No data loaded from {PHRASES_CSV_PATH}. Cannot proceed.")
        
    documents = _create_documents_from_dataframe(df)
    if not documents:
         raise ValueError("No documents were created from the DataFrame. Cannot proceed.")

    vector_store = _create_vector_store(documents, OPENAI_API_KEY)
    
    logger.info("Data loading and vector store creation complete.")
    return vector_store, df