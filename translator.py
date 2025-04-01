"""
Argentinian Spanish Translator Module

This module provides core translation functionality using LangChain and vector search
to find relevant Argentinian Spanish phrases and generate contextually appropriate translations.
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import pandas as pd
from prompts.manager import PromptManager


class ArgentinianTranslator:
    """
    Service for translating text to authentic Argentinian Spanish.
    
    This class uses a vector store of Argentinian phrases and expressions to
    provide context-aware translations that incorporate regional slang and expressions.
    """
    
    def __init__(self, vector_store, reference_df: pd.DataFrame, llm=None):
        """
        Initialize the translator with a vector store and reference data.
        
        Args:
            vector_store: FAISS vector store containing Argentinian phrases
            reference_df: Pandas DataFrame with the original phrases data
            llm: Optional language model (defaults to ChatOpenAI)
        """
        self.vector_store = vector_store
        self.reference_df = reference_df
        self.llm = llm or ChatOpenAI(temperature=0.7)
        self.prompt_manager = PromptManager()
        self.setup_translation_chain()
        
    def setup_translation_chain(self):
        """
        Set up the LangChain translation chain with a prompt template.
        
        This configures the prompt to instruct the model to translate text
        into authentic Argentinian Spanish using appropriate slang and expressions.
        """
        self.prompt = PromptTemplate(
            template=self.prompt_manager.translation_prompt,
            input_variables=["text", "reference_phrases"]
        )
        
        # Create a runnable sequence with the template and LLM
        self.chain = self.prompt | self.llm
    
    def get_relevant_phrases(self, input_text, n=3):
        """
        Retrieve relevant Argentinian phrases for the input text.
        
        Args:
            input_text: The text to find relevant phrases for
            n: Number of phrases to retrieve (default: 3)
            
        Returns:
            String containing the most relevant phrases from the vector store
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": n})
        docs = retriever.get_relevant_documents(input_text)
        
        if not docs:
            return "No specific Argentinian expressions found."
            
        phrases = "\n".join([doc.page_content for doc in docs])
        return phrases
    
    async def translate(self, input_text):
        """
        Translate input text to Argentinian Spanish asynchronously.
        
        Args:
            input_text: The text to translate
            
        Returns:
            Dictionary with original text, Argentinian translation, and reference phrases
        """
        # Get relevant reference phrases
        reference_phrases = self.get_relevant_phrases(input_text)
        
        # Run the translation chain using the correct async method
        result = await self.chain.ainvoke(
            {
                "text": input_text,
                "reference_phrases": reference_phrases
            }
        )
        
        return {
            "original_text": input_text,
            "argentinian_translation": result.content,
            "reference_phrases": reference_phrases
        }