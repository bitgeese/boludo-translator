"""
Custom Exception Classes for the Application
"""

class AppError(Exception):
    """Base class for application-specific errors."""
    pass

class ConfigError(AppError):
    """Error related to application configuration."""
    pass

class DataLoaderError(AppError):
    """Error occurring during data loading or vector store creation."""
    pass

class PromptError(AppError):
     """Error occurring during prompt loading or management."""
     pass

class TranslationError(AppError):
    """Error occurring during the translation process (e.g., LLM interaction)."""
    pass 