"""
Argentinian Spanish Translator Core Module

This module defines the ArgentinianTranslator class, responsible for the core
translation logic using a LangChain RAG (Retrieval-Augmented Generation) chain.
"""

import logging
from operator import itemgetter
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Assuming PromptManager is correctly placed and importable after restructuring
# If PromptManager was moved to core/, update the import accordingly.
# from core.prompt_manager import PromptManager
from core.prompt_manager import PromptManager 
from config import OPENAI_API_KEY, TRANSLATOR_MODEL_NAME

logger = logging.getLogger(__name__)


class ArgentinianTranslator:
    """
    Service for translating text to authentic Argentinian Spanish using RAG.

    This class uses a vector store of Argentinian phrases to retrieve relevant context
    and combines it with the input text in a prompt for an LLM to generate
    context-aware translations.
    """

    def __init__(
        self,
        vector_store: FAISS,
        prompt_manager: PromptManager,
        llm: BaseChatModel = None,
    ):
        """
        Initialize the translator.

        Args:
            vector_store: Initialized FAISS vector store for phrase retrieval.
            prompt_manager: Initialized PromptManager to load prompt templates.
            llm: Optional LangChain BaseChatModel instance. If None, initializes
                 ChatOpenAI with settings from config.
        """
        self.vector_store = vector_store
        self.prompt_manager = prompt_manager

        if llm:
            self.llm = llm
        else:
            logger.info(f"Initializing ChatOpenAI model: {TRANSLATOR_MODEL_NAME}")
            self.llm = ChatOpenAI(
                model=TRANSLATOR_MODEL_NAME,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.7, # Keep existing temperature
                streaming=True, # Enable streaming by default if needed later
            )

        self.retriever = self._create_retriever()
        self.chain = self._build_rag_chain()
        logger.info("ArgentinianTranslator initialized successfully.")

    def _create_retriever(self, k: int = 3):
        """Creates a retriever from the vector store."""
        logger.debug(f"Creating retriever with k={k}")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a string for the prompt context."""
        if not docs:
            return "No specific Argentinian expressions found as reference."
        # Using a clear separator for readability in the prompt
        return "\n---\n".join([doc.page_content.strip() for doc in docs])

    def _build_rag_chain(self):
        """Builds the LangChain Expression Language (LCEL) RAG chain."""
        logger.debug("Building RAG chain...")
        try:
            translation_prompt_template = ChatPromptTemplate.from_template(
                self.prompt_manager.translation_prompt
            )
            logger.debug("Loaded translation prompt template.")
        except AttributeError:
             logger.error("Failed to load translation_prompt from PromptManager.")
             raise ValueError("Translation prompt template not found in PromptManager.")
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise

        # LCEL Chain Definition
        rag_chain = (
            RunnablePassthrough.assign(
                # Retrieve documents based on the input text ("text")
                retrieved_docs=itemgetter("text") | self.retriever
            ).assign(
                # Format the retrieved documents into the "reference_phrases" context variable
                reference_phrases=itemgetter("retrieved_docs") | RunnableLambda(self._format_retrieved_docs)
            )
            # Prepare the input for the prompt template (needs "text" and "reference_phrases")
            | RunnablePassthrough.assign( # Ensure original 'text' is still available
                text=itemgetter("text")
            )
            | translation_prompt_template # Apply the prompt template
            | self.llm                   # Call the language model
            | StrOutputParser()         # Parse the output to a string
        )
        logger.info("RAG chain built successfully.")
        return rag_chain

    async def translate(self, input_text: str) -> str:
        """
        Translate input text to Argentinian Spanish using the RAG chain.

        Args:
            input_text: The text to translate.

        Returns:
            The translated text as a string.
        
        Raises:
            Exception: If the translation chain fails.
        """
        if not input_text:
            logger.warning("Translate called with empty input text.")
            return ""
            
        logger.info(f"Translating text: '{input_text[:50]}...'" ) 
        try:
            # The chain expects a dictionary with the key "text"
            result = await self.chain.ainvoke({"text": input_text})
            logger.info("Translation successful.")
            return result
        except Exception as e:
            logger.error(f"Error during translation chain invocation: {e}", exc_info=True)
            # Depending on requirements, could return an error message or raise
            raise RuntimeError(f"Translation failed: {e}") from e