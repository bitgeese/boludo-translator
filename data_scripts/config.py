"""
Configuration module for Argentine Spanish Learning RAG data processing scripts.

This module centralizes all configuration values for the data processing pipeline,
providing default values and allowing overrides via environment variables.
This approach eliminates hardcoded constants and improves flexibility
and maintainability.

Key configuration areas:
- File paths for input/output files
- Text processing parameters (chunk size, overlap)
- CSV column configurations
- Scraping parameters (delay, timeout)
- HTTP request headers
- Logging settings

Usage examples:
    from data_scripts.config import CHUNK_SIZE, CSV_INPUT_FILE, log_config

    # Log current configuration
    log_config()

    # Use configuration values
    chunks = split_text(text, chunk_size=CHUNK_SIZE)
    data = load_csv(CSV_INPUT_FILE)
"""

import logging
import os
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Script input/output file paths
JSONL_INPUT_FILE = os.environ.get("JSONL_INPUT_FILE", "ventureout_data.jsonl")
CSV_INPUT_FILE = os.environ.get("CSV_INPUT_FILE", "argentine_spanish_qa.csv")
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "../chroma_db")
OUTPUT_JSONL_FILE = os.environ.get("OUTPUT_JSONL_FILE", "ventureout_data.jsonl")

# Text processing settings
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MIN_CONTENT_LENGTH = int(os.environ.get("MIN_CONTENT_LENGTH", "100"))

# CSV column configuration
CSV_TEXT_COLUMN = os.environ.get("CSV_TEXT_COLUMN", "Answer")
CSV_METADATA_COLUMNS = os.environ.get("CSV_METADATA_COLUMNS", "Question,Source").split(
    ","
)

# Scraping configuration
SITEMAP_INDEX_URL = os.environ.get(
    "SITEMAP_INDEX_URL", "https://ventureoutspanish.com/sitemap_index.xml"
)
REQUEST_DELAY = float(
    os.environ.get("REQUEST_DELAY", "1.0")
)  # Delay in seconds between requests


# Standard browser headers to avoid 406 errors
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# Get custom headers from environment (useful for testing or working around blocks)
def get_request_headers() -> Dict[str, str]:
    """
    Get request headers for HTTP requests, with environment variable overrides if set.

    Returns:
        Dictionary of HTTP headers
    """
    headers = DEFAULT_HEADERS.copy()

    # Check if any header is defined in environment variables
    custom_user_agent = os.environ.get("HTTP_USER_AGENT")
    if custom_user_agent:
        headers["User-Agent"] = custom_user_agent

    custom_accept = os.environ.get("HTTP_ACCEPT")
    if custom_accept:
        headers["Accept"] = custom_accept

    return headers


def get_script_paths() -> Dict[str, Path]:
    """
    Get absolute paths for script input and output files.

    Returns:
        Dictionary of file paths
    """
    # Get the directory where the data_scripts folder is located
    base_dir = Path(__file__).parent

    return {
        "jsonl_input": base_dir / JSONL_INPUT_FILE,
        "csv_input": base_dir / CSV_INPUT_FILE,
        "chroma_dir": base_dir.parent
        / Path(CHROMA_PERSIST_DIRECTORY).relative_to("../")
        if CHROMA_PERSIST_DIRECTORY.startswith("../")
        else base_dir / CHROMA_PERSIST_DIRECTORY,
        "output_jsonl": base_dir / OUTPUT_JSONL_FILE,
    }


def log_config() -> None:
    """Log the current configuration values."""
    config = {
        "JSONL_INPUT_FILE": JSONL_INPUT_FILE,
        "CSV_INPUT_FILE": CSV_INPUT_FILE,
        "CHROMA_PERSIST_DIRECTORY": CHROMA_PERSIST_DIRECTORY,
        "OUTPUT_JSONL_FILE": OUTPUT_JSONL_FILE,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "MIN_CONTENT_LENGTH": MIN_CONTENT_LENGTH,
        "CSV_TEXT_COLUMN": CSV_TEXT_COLUMN,
        "CSV_METADATA_COLUMNS": CSV_METADATA_COLUMNS,
        "SITEMAP_INDEX_URL": SITEMAP_INDEX_URL,
        "REQUEST_DELAY": REQUEST_DELAY,
    }

    logger.info("Script configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
