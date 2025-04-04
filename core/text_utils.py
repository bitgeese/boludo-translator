"""
Text Utility Module for Argentine Spanish Learning RAG System

This module provides common text processing and cleaning utilities
that are shared across the application and data processing scripts.

Primary Functions:
- Cleaning web-scraped content by removing boilerplate elements
- Identifying and trimming content at common ending points
- Removing excessive whitespace and formatting issues
- Handling category listings and other non-content elements

The module is designed to work with content from WordPress-based blogs,
particularly the VentureOut Spanish blog, but can be customized for other
sources by adding pattern definitions.

Usage:
    from core.text_utils import clean_text, add_custom_patterns

    # Basic text cleaning
    cleaned_content = clean_text(raw_content)

    # Add custom patterns for special cases
    add_custom_patterns([r"My custom pattern.*?to remove"])

    # For performance with large datasets
    compiled_patterns = compile_patterns()
    cleaned_content = clean_text_with_compiled_patterns(raw_content, compiled_patterns)
"""

import logging
import re
from typing import List, Pattern

logger = logging.getLogger(__name__)

# Common patterns to remove from the text content for cleaning
# These patterns identify boilerplate content commonly found on WordPress sites
PATTERNS_TO_REMOVE = [
    # Comment form and related elements
    r"Leave a Reply\s*Cancel reply.*?for the next time I comment\.",
    r"Comment\s*Enter your name.*?for the next time I comment\.",
    r"Enter your name or username to comment.*?for the next time I comment\.",
    # WordPress footer elements and metadata
    r"Copyright © \d{4}.*?All rights reserved\.?",
    r"Post author:\s*.*?\s*Post published:\s*.*?\s*Reading time:\s*.*?\s*",
    r"Thank you for sharing this post!\s*Share this content\s*Opens in a new window\s*",
    # Navigation elements
    r"Previous Post.*?Next Post",
    # Social sharing
    r"Share this:.*?Click to share",
    r"Share this content.*?Opens in a new window",
    # Search prompts
    r"Search for:.*?search",
    # Category and tag listings at the end of posts
    r"Categories.*?\(\d+\).*?Spanish Teaching.*?\(\d+\)",
    r"\(\d+\)\s*Argentinian Spanish\s*\(\d+\)\s*Argentinian Spanish Curse Words"
    r"\s*\(\d+\)",
    # Common footer text
    r"Venture Out Spanish\s*·\s*\S+\s*\S+\s*·\s*Privacy Policy",
    # Author bio section
    r"About the author.*?View all posts",
    # Comment section headers
    r"Comments\s*\(\d+\)",
    # WordPress tags
    r"Filed under:.*?Tags:",
    # Post metadata block
    r"Post author:.*?Reading time:.*?read",
    # Share buttons
    r"Opens in a new window\s*Opens in a new window\s*Opens in a new window",
]

# Common ending points for content where boilerplate typically begins
SPLIT_POINTS = [
    "Leave a Reply",
    "Related Posts",
    "Categories",
    "Thank you for sharing this post",
    "Post navigation",
]


def clean_text(text: str, min_content_length: int = 10) -> str:
    """
    Clean up text content by removing common boilerplate elements.

    Args:
        text: The text content to clean
        min_content_length: Minimum length for valid content (after cleaning)

    Returns:
        Cleaned text with boilerplate content removed
    """
    if not text:
        logger.warning("Empty text provided for cleaning")
        return ""

    # First, remove any identified patterns
    for pattern in PATTERNS_TO_REMOVE:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Split text at common ending points
    for point in SPLIT_POINTS:
        if point in text:
            text = text.split(point)[0]

    # Remove category listings (common at the end of posts)
    lines = text.splitlines()
    filtered_lines = []
    skip_mode = False
    category_pattern = re.compile(r"^\s*\(\d+\)\s*$")

    for line in lines:
        # If line is like "(45)" - part of category listings - enter skip mode
        if category_pattern.match(line):
            skip_mode = True

        # If we're not in skip mode, keep the line
        if not skip_mode:
            filtered_lines.append(line)

        # If we encounter a long line after categories, exit skip mode
        if skip_mode and len(line.strip()) > 30:
            skip_mode = False

    # Rejoin the filtered lines
    text = "\n".join(filtered_lines)

    # Handle missing content
    if not text or len(text.strip()) < min_content_length:
        logger.debug(
            f"Text content too short after cleaning: {len(text.strip())} chars"
        )
        return "No usable content found."

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # Replace 3+ newlines with 2
    text = re.sub(r"\s{2,}", " ", text)  # Replace 2+ spaces with 1

    # Remove any URLs that might be in the text
    text = re.sub(r"https?://\S+", "", text)

    return text.strip()


def add_custom_patterns(patterns: List[str]) -> None:
    """
    Add custom patterns to the global PATTERNS_TO_REMOVE list.

    Args:
        patterns: List of regex patterns to add
    """
    global PATTERNS_TO_REMOVE
    PATTERNS_TO_REMOVE.extend(patterns)
    logger.debug(f"Added {len(patterns)} custom patterns to removal list")


def compile_patterns() -> List[Pattern]:
    """
    Precompile all regex patterns for improved performance.

    Returns:
        List of compiled regex patterns
    """
    compiled_patterns = []
    for pattern in PATTERNS_TO_REMOVE:
        try:
            compiled_patterns.append(
                re.compile(pattern, flags=re.DOTALL | re.IGNORECASE)
            )
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern} - Error: {e}")

    logger.debug(f"Compiled {len(compiled_patterns)} regex patterns")
    return compiled_patterns


def clean_text_with_compiled_patterns(
    text: str, compiled_patterns: List[Pattern]
) -> str:
    """
    Clean text using precompiled patterns for better performance on large datasets.

    Args:
        text: The text content to clean
        compiled_patterns: List of precompiled regex patterns

    Returns:
        Cleaned text with boilerplate content removed
    """
    if not text:
        return ""

    # Apply each compiled pattern
    for pattern in compiled_patterns:
        text = pattern.sub("", text)

    # Continue with the rest of the cleaning process
    # (This is a simplified version - full implementation would duplicate the
    # remaining cleaning steps from clean_text())

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()
