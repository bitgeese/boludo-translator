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
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- OpenAI Configuration ---
    OPENAI_API_KEY: str

    # --- Data Configuration ---
    PHRASES_CSV_PATH: str = "data/phrases.csv"

    # --- Prompt Configuration ---
    PROMPTS_DIR: str = "prompts"
    SYSTEM_PROMPT_FILE: str = "system.md"
    TRANSLATION_PROMPT_FILE: str = "translation.md"

    # --- Translator Configuration ---
    TRANSLATOR_MODEL_NAME: str = "gpt-4o-mini"
    TRANSLATOR_TEMPERATURE: float = 0.7

    # --- Logging --- 
    LOG_LEVEL: str = "INFO"

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