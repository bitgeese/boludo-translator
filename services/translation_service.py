"""
Translation Service Module

Provides a high-level interface for accessing translation functionality.
It initializes and holds the core ArgentinianTranslator instance.
"""

import logging
import pandas as pd
from langchain_community.vectorstores import FAISS

from core.translator import ArgentinianTranslator
from core.prompt_manager import PromptManager
from core.exceptions import TranslationError, AppError

logger = logging.getLogger(__name__)

class TranslationService:
    """Orchestrates translation using the ArgentinianTranslator."""

    def __init__(self, vector_store: FAISS, prompt_manager: PromptManager):
        """
        Initializes the TranslationService.

        Args:
            vector_store: The pre-loaded FAISS vector store.
            prompt_manager: The pre-initialized PromptManager.
        """
        if not vector_store:
            logger.error("TranslationService requires a valid vector_store.")
            raise ValueError("vector_store cannot be None")
        if not prompt_manager:
             logger.error("TranslationService requires a valid prompt_manager.")
             raise ValueError("prompt_manager cannot be None")
             
        self.translator = ArgentinianTranslator(
            vector_store=vector_store,
            prompt_manager=prompt_manager
            # LLM will be initialized internally by ArgentinianTranslator using config
        )
        logger.info("TranslationService initialized successfully.")

    async def translate_text(self, text: str) -> str:
        """
        Translates the given text using the underlying translator.

        Args:
            text: The input text to translate.

        Returns:
            The translated text.
            
        Raises:
            RuntimeError: If the translation fails.
        """
        logger.debug(f"TranslationService received request to translate: '{text[:50]}...'")
        try:
            translated_text = await self.translator.translate(text)
            return translated_text
        except Exception as e:
            # Log the error originating from the translator
            logger.error(f"TranslationService encountered an error during translation: {e}", exc_info=True)
            # Re-raise or handle as appropriate for the application layer
            # Wrap core errors in a service-level error if desired, or just re-raise
            if isinstance(e, TranslationError):
                raise # Re-raise the specific error
            else:
                # Wrap unexpected errors from the core layer or dependencies
                raise AppError("An unexpected error occurred in the translation core.") from e 