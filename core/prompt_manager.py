"""
Prompt Manager Module

Handles loading prompt templates from markdown files located in the directory
specified in the application's configuration.
"""

import logging
from pathlib import Path

# Import configuration constants
# from config import PROMPTS_DIR, SYSTEM_PROMPT_FILE, TRANSLATION_PROMPT_FILE
# Import the settings object instead
from config import settings
# Import custom exception
from .exceptions import PromptError

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and accessing prompts from markdown files."""

    def __init__(self):
        """
        Initialize the prompt manager using the directory specified in the config.
        """
        self.prompts_dir = Path(settings.PROMPTS_DIR)
        self._ensure_prompts_dir_exists()
        logger.info(f"PromptManager initialized. Using prompt directory: {self.prompts_dir}")
        # Load prompts on initialization to fail fast if they are missing
        try:
            self._system_prompt = self._load_prompt(settings.SYSTEM_PROMPT_FILE)
            self._translation_prompt = self._load_prompt(settings.TRANSLATION_PROMPT_FILE)
            logger.info("Successfully loaded system and translation prompts.")
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize PromptManager: {e}", exc_info=True)
            # Wrap in custom exception
            raise PromptError(f"Missing required prompt file: {e}") from e
        except IOError as e:
            logger.error(f"Failed to initialize PromptManager due to IO error: {e}", exc_info=True)
            raise PromptError(f"IO error loading prompt file: {e}") from e
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error initializing PromptManager: {e}", exc_info=True)
            raise PromptError(f"Unexpected error initializing prompts: {e}") from e

    def _ensure_prompts_dir_exists(self):
        """Checks if the prompts directory exists."""
        if not self.prompts_dir.is_dir():
            logger.error(f"Prompt directory not found: {self.prompts_dir}")
            # Raise specific error
            raise PromptError(f"Prompt directory not found: {self.prompts_dir}")

    def _load_prompt(self, filename: str) -> str:
        """
        Load a prompt from a specific file within the configured prompts directory.

        Args:
            filename: Full name of the markdown file (e.g., 'system.md')

        Returns:
            The prompt content as a string.

        Raises:
            PromptError: If the file is not found or cannot be read.
        """
        file_path = self.prompts_dir / filename
        logger.debug(f"Attempting to load prompt from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Prompt file '{filename}' is empty.")
                return content
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {file_path}")
            # Raise specific error
            raise PromptError(f"Prompt file not found: {file_path}") from None
        except IOError as e:
            logger.error(f"Error reading prompt file {file_path}: {e}")
            # Raise specific error
            raise PromptError(f"Error reading prompt file {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error reading prompt file {file_path}: {e}", exc_info=True)
            raise PromptError(f"Unexpected error reading prompt file {file_path}: {e}") from e

    @property
    def system_prompt(self) -> str:
        """Returns the loaded system prompt."""
        return self._system_prompt

    @property
    def translation_prompt(self) -> str:
        """Returns the loaded translation prompt template."""
        return self._translation_prompt 