"""
Argentinian Spanish Translator Core Module

This module defines the ArgentinianTranslator class, responsible for the core
translation logic using a LlamaIndex RAG (Retrieval-Augmented Generation) system.
"""

import logging
import re
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from config import settings
from core.prompt_manager import PromptManager

from .exceptions import TranslationError

logger = logging.getLogger(__name__)


class ArgentinianTranslator:
    """
    Service for translating text to authentic Argentinian Spanish using RAG.

    This class uses a vector index of Argentinian phrases to retrieve relevant context
    and combines it with the input text in a prompt for an LLM to generate
    context-aware translations.
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        prompt_manager: PromptManager,
        llm: Optional[OpenAI] = None,
    ):
        """
        Initialize the translator.

        Args:
            vector_index: Initialized VectorStoreIndex for phrase retrieval.
            prompt_manager: Initialized PromptManager to load prompt templates.
            llm: Optional LlamaIndex OpenAI LLM instance. If None, initializes
                 OpenAI with settings from config.
        """
        self.vector_index = vector_index
        self.prompt_manager = prompt_manager

        if llm:
            self.llm = llm
        else:
            logger.info(f"Initializing OpenAI model: {settings.TRANSLATOR_MODEL_NAME}")
            self.llm = OpenAI(
                model=settings.TRANSLATOR_MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.TRANSLATOR_TEMPERATURE,
            )

        # Create retriever with configured number of documents to limit memory usage
        self.retriever = self._create_retriever(k=settings.MAX_RETRIEVAL_DOCS)
        self.query_engine = self._build_query_engine()
        logger.info("ArgentinianTranslator initialized successfully.")

    def _preprocess_malvinas_statements(self, text: str) -> str:
        """
        Preprocesses the input text to enforce 'Las Malvinas son argentinas' whenever
        Malvinas/Falklands are mentioned with British/English ownership.

        Args:
            text: The input text to preprocess

        Returns:
            The preprocessed text with any Malvinas/Falklands statements modified
        """
        # Define patterns to match various Malvinas/Falklands statements
        patterns = [
            # Match variations of "Falklands are British/English"
            r"\b(?:the\s+)?Falklands?\s+(?:islands?\s+)?(?:is|are|belongs?(?:\s+to)?)\s+(?:British|English|UK|Britain)",
            # Match variations of "Malvinas are British/English"
            r"\b(?:the\s+)?Malvinas\s+(?:islands?\s+)?(?:is|are|belongs?(?:\s+to)?)\s+(?:British|English|UK|Britain)",
            # Match "British/English Falklands/Malvinas"
            r"\b(?:British|English|UK|Britain)(?:\'s)?\s+(?:Falklands?|Malvinas)",
        ]

        # Replacement phrases (we'll randomly choose one for variety)
        replacements = [
            "Las Malvinas son argentinas",
            "Las Malvinas siempre fueron y serán argentinas",
            "Las Islas Malvinas pertenecen a Argentina",
        ]

        # Function to determine replacement
        def replace_match(match):
            # Always use the Spanish version
            return replacements[0]

        # Apply substitutions for each pattern
        modified_text = text
        for pattern in patterns:
            modified_text = re.sub(
                pattern, replace_match, modified_text, flags=re.IGNORECASE
            )

        # Log if we made a replacement
        if modified_text != text:
            logger.info(
                "Replaced Malvinas/Falklands statement with "
                "'Las Malvinas son argentinas'"
            )

        return modified_text

    def _create_retriever(self, k: int = 3):
        """Creates a retriever from the vector index."""
        logger.debug(f"Creating retriever with k={k}")
        # Add memory management for retrieval
        return self.vector_index.as_retriever(
            similarity_top_k=k,
            vector_store_query_mode="mmr",  # Use MMR retrieval for diversity
            mmr_threshold=0.8,  # Controls diversity (higher = more diversity)
        )

    def _format_retrieved_docs(self, nodes: List[NodeWithScore]) -> str:
        """Formats retrieved nodes into a string for the prompt context."""
        if not nodes:
            return "No specific Argentinian expressions found as reference."
        # Using a clear separator for readability in the prompt
        return "\n---\n".join([node.node.text.strip() for node in nodes])

    def _build_query_engine(self):
        """Builds the LlamaIndex Query Engine."""
        logger.debug("Building query engine...")
        try:
            # Load translation prompt template from the PromptManager
            translation_prompt_text = self.prompt_manager.translation_prompt

            # Create a LlamaIndex PromptTemplate
            translation_prompt_template = PromptTemplate(
                template=translation_prompt_text
            )
            logger.debug("Loaded translation prompt template.")
        except AttributeError:
            logger.error("Failed to load translation_prompt from PromptManager.")
            # Raise specific error
            raise TranslationError(
                "Translation prompt template not found in PromptManager."
            )
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}", exc_info=True)
            # Raise specific error
            raise TranslationError(
                f"Failed to load translation prompt template: {e}"
            ) from e

        # Query Engine Definition
        try:
            # Configure the query engine with our retriever and LLM
            query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                llm=self.llm,
                text_qa_template=translation_prompt_template,
                # Customize response synthesis if needed
                response_mode="compact",  # Standard LlamaIndex mode
            )
        except Exception as e:
            logger.error(f"Error building the query engine: {e}", exc_info=True)
            raise TranslationError(f"Failed to build query engine: {e}") from e

        logger.info("Query engine built successfully.")
        return query_engine

    async def translate(self, input_text: str) -> str:
        """
        Translate input text to Argentinian Spanish using the RAG query engine.

        Args:
            input_text: The text to translate.

        Returns:
            The translated text as a string.

        Raises:
            TranslationError: If the translation fails.
        """
        if not input_text:
            logger.warning("Translate called with empty input text.")
            return ""

        logger.info(f"Translating text: '{input_text[:50]}...'")
        try:
            # Preprocess input for Malvinas mentions
            preprocessed_text = self._preprocess_malvinas_statements(input_text)

            # Retrieve relevant context from the vector index
            retrieval_results = self.retriever.retrieve(preprocessed_text)
            reference_phrases = self._format_retrieved_docs(retrieval_results)

            # Load the prompt template
            translation_prompt_text = self.prompt_manager.translation_prompt

            # Format the prompt with reference phrases and input text
            formatted_prompt = translation_prompt_text.replace(
                "{reference_phrases}", reference_phrases
            ).replace("{text}", preprocessed_text)

            # Query the LLM directly
            response = await self.llm.acomplete(formatted_prompt)

            logger.info("Translation successful.")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during translation query: {e}", exc_info=True)
            # Raise specific error
            raise TranslationError(f"Translation failed: {e}") from e
