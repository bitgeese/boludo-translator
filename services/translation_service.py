"""
Translation Service Module

Provides a high-level interface for accessing translation functionality.
It initializes and holds the core ArgentinianTranslator instance.
"""

import logging
import re  # For language code validation
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langdetect import LangDetectException, detect
from langdetect.detector_factory import DetectorFactory

from config import settings  # Import settings
from core.exceptions import AppError, TranslationError
from core.prompt_manager import PromptManager
from core.translator import ArgentinianTranslator

logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_LANGS = ["en", "es"]
UNKNOWN_LANG_CODE = "unknown"  # Standardize unknown code
# Simple pattern for 2-letter ISO codes
ISO_639_1_PATTERN = re.compile(r"^[a-z]{2}$")

# --- Language Detection Prompt Template (Keep it minimal) ---
# Use a single triple-quoted string for clarity and correct parsing
LANG_DETECT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    f"""Identify the primary language of the following text. \
Respond with ONLY the two-letter ISO 639-1 language code (e.g., 'en', 'es', 'fr'). \
If you are unsure, the text is nonsensical, gibberish, or not a real language, \
respond with '{UNKNOWN_LANG_CODE}'. \
Text:
\"\"\"
{{user_input}}
\"\"\"
Language code:"""
)

# Seed langdetect for consistent results
try:
    DetectorFactory.seed = 0
    logger.info("Language detector seeded.")
except NameError:
    logger.warning("Could not seed DetectorFactory for langdetect.")
    pass


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
            vector_store=vector_store, prompt_manager=prompt_manager
        )

        # Create the LLM language detection chain (reuses the translator's LLM)
        self._lang_detect_chain = (
            LANG_DETECT_PROMPT_TEMPLATE | self.translator.llm | StrOutputParser()
        )
        logger.info("LLM language detection chain created.")

        logger.info("TranslationService initialized successfully.")

    async def _detect_language_statistical(self, text: str) -> str:
        """Detect language using langdetect (statistical)."""
        try:
            detected_lang = detect(text)
            logger.debug(f"Statistical detection result: {detected_lang}")
            # Basic validation just in case langdetect returns something odd
            if ISO_639_1_PATTERN.match(detected_lang):
                return detected_lang.lower()
            else:
                logger.warning(
                    f"Statistical detection returned non-ISO format: {detected_lang}. "
                    f"Treating as unknown."
                )
                return UNKNOWN_LANG_CODE
        except LangDetectException:
            logger.warning(
                f"Statistical detection failed for: '{text[:50]}...'.", exc_info=False
            )
            return UNKNOWN_LANG_CODE
        except Exception as e:
            logger.error(
                f"Unexpected error during statistical language detection: {e}",
                exc_info=True,
            )
            # Don't raise AppError here, just return unknown to allow
            # potential translation
            return UNKNOWN_LANG_CODE

    async def _detect_language_llm(self, text: str) -> str:
        """Detect language using the LLM chain."""
        logger.debug(f"Using LLM for language detection for: '{text[:50]}...'")
        try:
            # Use ainvoke for the async context
            result = await self._lang_detect_chain.ainvoke({"user_input": text})
            # Clean up potential whitespace and normalize case
            detected_lang = result.strip().lower()
            logger.debug(f"LLM detection result: {detected_lang}")

            # Validate the output format
            if detected_lang == UNKNOWN_LANG_CODE or ISO_639_1_PATTERN.match(
                detected_lang
            ):
                return detected_lang
            else:
                logger.warning(
                    f"LLM detection returned unexpected format: '{result}'. "
                    f"Treating as unknown."
                )
                return UNKNOWN_LANG_CODE

        except Exception as e:
            logger.error(f"Error during LLM language detection: {e}", exc_info=True)
            # Gracefully handle LLM errors by returning unknown, allowing
            # potential translation
            return UNKNOWN_LANG_CODE

    async def _detect_language(self, text: str) -> str:
        """Detects language using hybrid approach: LLM for short, statistical
        for long."""
        word_count = len(text.split())  # Simple word count
        logger.debug(f"Input word count: {word_count}")

        if word_count <= settings.SHORT_INPUT_WORD_THRESHOLD:
            logger.debug("Input is short, using LLM detection.")
            return await self._detect_language_llm(text)
        else:
            logger.debug("Input is long, using statistical detection.")
            return await self._detect_language_statistical(text)

    async def translate_text(
        self, text: str, config: Optional[RunnableConfig] = None
    ) -> str:
        """
        Detects the language of the text using a hybrid approach and translates
        it if it's English or Spanish.

        Args:
            text: The input text to translate.
            config: Optional RunnableConfig to pass down to the translator's
                    translate method (e.g., for callbacks).

        Returns:
            The translated text, or a message indicating the language is unsupported.

        Raises:
            AppError: If an unexpected error occurs during detection or
                      translation setup.
            TranslationError: If the core translation fails.
        """
        logger.debug(
            f"TranslationService received request: '{text[:50]}...' "
            f"with config: {config}"
        )

        if not text.strip():
            logger.warning("Received empty or whitespace-only text.")
            return "Please provide some text to translate."

        # 1. Language Detection (Hybrid Approach)
        try:
            detected_lang = await self._detect_language(text)
            logger.info(f"Detected language (hybrid): {detected_lang}")
        except Exception as e:
            # This catch block is primarily for unexpected errors within
            # _detect_language itself, though the sub-methods handle their
            # specific errors more gracefully.
            logger.error(
                f"Unexpected error during hybrid language detection orchestration: {e}",
                exc_info=True,
            )
            raise AppError(
                "An unexpected error occurred during language detection setup."
            ) from e

        # 2. Check if language is supported
        if detected_lang not in SUPPORTED_LANGS and detected_lang != UNKNOWN_LANG_CODE:
            logger.info(
                f"Unsupported language detected ({detected_lang}). "
                f"Skipping translation."
            )
            # Provide a user-friendly message
            return (
                f"Sorry, I currently only support English ('en') and Spanish ('es'). "
                f"Detected: {detected_lang}"
            )

        # 3. Proceed with Translation if supported or unknown
        if detected_lang == UNKNOWN_LANG_CODE:
            logger.info("Language is unknown, proceeding with translation attempt.")
        else:
            logger.info(
                f"Language '{detected_lang}' is supported, proceeding with translation."
            )

        try:
            translated_text = await self.translator.translate(text, config=config)
            return translated_text
        except TranslationError as e:
            logger.error(
                f"TranslationService re-raising TranslationError: {e}", exc_info=False
            )
            raise e  # Re-raise specific TranslationError to be handled by UI
        except Exception as e:
            logger.error(
                f"TranslationService encountered an unexpected error during "
                f"core translation: {e}",
                exc_info=True,
            )
            raise AppError(
                "An unexpected error occurred during the translation process."
            ) from e
