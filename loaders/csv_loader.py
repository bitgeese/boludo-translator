"""
Argentinian Spanish CSV Loader Module

This module provides functionality to load and process CSV data containing
Argentinian Spanish phrases and expressions, converting them into searchable
vector embeddings for semantic retrieval.
"""

import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict


class ArgentinianSpanishLoader:
    """
    Loader for Argentinian Spanish CSV reference data.
    
    This class handles loading data from a CSV file containing Argentinian Spanish
    phrases and expressions, converting them to LangChain Document objects, and
    creating a vector store for semantic search.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the loader with a path to the CSV file.
        
        Args:
            file_path: Path to the CSV file containing Argentinian Spanish phrases
        """
        self.file_path = file_path
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data into a pandas DataFrame.
        
        Returns:
            DataFrame containing the loaded CSV data
            
        Raises:
            Exception: If the file cannot be read or is not in the expected format
        """
        try:
            df = pd.read_csv(self.file_path)
            print(f"Loaded {len(df)} phrases from {self.file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame rows to LangChain Document objects.
        
        This method transforms each row in the DataFrame into a Document object
        with structured content and metadata for better retrieval.
        
        Args:
            df: DataFrame containing the Argentinian Spanish phrases
            
        Returns:
            List of Document objects with phrase content and metadata
        """
        documents = []
        
        for _, row in df.iterrows():
            # Create content from row data
            content = f"""
            Original: {row['Original Phrase/Word']}
            Argentinian: {row['Argentinian Equivalent']}
            Context: {row['Explanation (Context/Usage)']}
            Region: {row['Region Specificity']}
            Formality: {row['Level of Formality']}
            """
            
            # Create metadata for better retrieval
            metadata = {
                "original": row['Original Phrase/Word'],
                "argentinian": row['Argentinian Equivalent'],
                "context": row['Explanation (Context/Usage)'],
                "region": row['Region Specificity'],
                "formality": row['Level of Formality']
            }
            
            # Create a Document object
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
        return documents
    
    def create_vector_store(self, documents: List[Document], embedding_model=None):
        """
        Create a vector store from the documents for semantic search.
        
        This method creates a FAISS vector store using the provided documents,
        allowing for efficient semantic search of phrases.
        
        Args:
            documents: List of Document objects to index
            embedding_model: Optional embedding model (defaults to OpenAIEmbeddings)
            
        Returns:
            FAISS vector store containing the document embeddings
        """
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings()
            
        vector_store = FAISS.from_documents(documents, embedding_model)
        return vector_store
    
    def load_and_process(self):
        """
        Load, process and create a searchable vector store.
        
        This method combines all the steps needed to go from CSV file to
        searchable vector store and reference data.
        
        Returns:
            Tuple containing (vector_store, reference_dataframe)
        """
        df = self.load_data()
        documents = self.create_documents(df)
        vector_store = self.create_vector_store(documents)
        return vector_store, df