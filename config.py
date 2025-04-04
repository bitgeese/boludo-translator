"""
Application Configuration Module

Uses Pydantic's BaseSettings for typed configuration management,
loading values from environment variables and .env files.
"""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Load .env file besides environment variables
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI Configuration ---
    OPENAI_API_KEY: str

    # --- Data Configuration ---
    PHRASES_CSV_PATH: str = "data/phrases.csv"
    # New settings for VentureOut data
    VENTUREOUT_DATA_PATH: str = "data/ventureout_data.jsonl"
    USE_VENTUREOUT_DATA: bool = True

    # --- Vector Store Configuration ---
    VECTORSTORE_TYPE: str = "CHROMA"  # Options: FAISS or CHROMA
    # Chroma is set as default because:
    # 1. Provides persistent storage by default, preserving embeddings between restarts
    # 2. Better support for metadata filtering (by source, region, etc.)
    # 3. Dataset size is moderate (phrases + VentureOut content < 1MB)
    # 4. More user-friendly for development with automatic persistence
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    # --- Prompt Configuration ---
    PROMPTS_DIR: str = "prompts"
    SYSTEM_PROMPT_FILE: str = "system.md"
    TRANSLATION_PROMPT_FILE: str = "translation.md"

    # --- Translator Configuration ---
    TRANSLATOR_MODEL_NAME: str = "gpt-4o"
    TRANSLATOR_TEMPERATURE: float = 0.7

    # --- Language Detection Configuration ---
    SHORT_INPUT_WORD_THRESHOLD: int = 2  # Use LLM if word count <= this
    # Optional: Specify a different model just for detection if needed
    # LANGUAGE_DETECTION_MODEL_NAME: str | None = "gpt-3.5-turbo"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Debugging ---
    DEBUG: bool = (
        False  # Set to True via env var to enable debug features like step visibility
    )


# Instantiate settings. Pydantic automatically loads and validates.
try:
    settings = Settings()
    # You might want to log the loaded settings (excluding secrets)
    # logger.info(f"Loaded settings: {settings.model_dump(exclude={'OPENAI_API_KEY'})}")
except Exception as e:
    logger.critical(f"Failed to load application settings: {e}", exc_info=True)
    # Depending on the app, you might want to exit here
    raise SystemExit(f"Configuration error: {e}")

# Export the instantiated settings object for other modules to import
__all__ = ["settings"]
