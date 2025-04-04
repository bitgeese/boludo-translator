"""
Exceptions Module for Data Scripts

This module defines custom exceptions used throughout the data processing scripts.
Using custom exceptions allows for more specific error handling and clearer
error messages.
"""


class DataScriptError(Exception):
    """Base exception for all data script related errors."""

    pass


class ScrapingError(DataScriptError):
    """Raised when there is an error during web scraping."""

    pass


class XMLParsingError(ScrapingError):
    """Raised when there is an error parsing XML content."""

    pass


class SitemapError(ScrapingError):
    """Raised when there is an error processing a sitemap."""

    pass


class DataLoadingError(DataScriptError):
    """Raised when there is an error loading data from a file."""

    pass


class DataProcessingError(DataScriptError):
    """Raised when there is an error processing data."""

    pass


class EmbeddingError(DataScriptError):
    """Raised when there is an error creating embeddings."""

    pass


class VectorStoreError(DataScriptError):
    """Raised when there is an error with the vector store."""

    pass


class ConfigurationError(DataScriptError):
    """Raised when there is an error with the configuration."""

    pass
