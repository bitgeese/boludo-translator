# Standard library imports
import json
import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Third-party library imports
import requests
from bs4 import BeautifulSoup
from lxml import etree

# Local application imports
try:
    # Import custom exceptions
    # Import config
    from data_scripts.config import (
        OUTPUT_JSONL_FILE,
        REQUEST_DELAY,
        SITEMAP_INDEX_URL,
        get_request_headers,
        log_config,
    )
    from data_scripts.exceptions import (
        DataScriptError,
        ScrapingError,
        SitemapError,
        XMLParsingError,
    )
except ImportError:
    # Fallback exception classes if import fails
    class DataScriptError(Exception):
        pass

    class ScrapingError(DataScriptError):
        pass

    class XMLParsingError(ScrapingError):
        pass

    class SitemapError(ScrapingError):
        pass

    # Fallback configuration
    SITEMAP_INDEX_URL = "https://ventureoutspanish.com/sitemap_index.xml"
    OUTPUT_JSONL_FILE = "ventureout_data.jsonl"
    REQUEST_DELAY = 1.0

    # Fallback headers
    HEADERS = {
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

    def get_request_headers():
        return HEADERS

    def log_config():
        pass


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get request headers from config
HEADERS = get_request_headers()


def fetch_xml(url: str) -> Optional[etree._Element]:
    """
    Fetches and parses XML content from a URL.

    Args:
        url: The URL to fetch XML from

    Returns:
        Parsed XML element tree or None if fetch/parse fails

    Raises:
        XMLParsingError: If there's an error parsing the XML content
        ScrapingError: If there's an error fetching the content
    """
    try:
        logging.info(f"Fetching XML from {url}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Use lxml for robust XML parsing
        parser = etree.XMLParser(recover=True)
        try:
            tree = etree.fromstring(response.content, parser=parser)
            return tree
        except etree.XMLSyntaxError as e:
            error_msg = f"Error parsing XML from {url}: {e}"
            logging.error(error_msg)
            # Include first 100 chars of content in error for debugging
            content_preview = response.content[:100].decode("utf-8", errors="replace")
            raise XMLParsingError(
                f"{error_msg}\nContent preview: {content_preview}"
            ) from e

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching XML from {url}: {e}"
        logging.error(error_msg)
        raise ScrapingError(error_msg) from e
    except Exception as e:
        if isinstance(e, (XMLParsingError, ScrapingError)):
            # Re-raise our custom exceptions
            raise
        error_msg = f"Unexpected error fetching/parsing XML from {url}: {e}"
        logging.error(error_msg, exc_info=True)
        raise ScrapingError(error_msg) from e


def extract_urls_from_sitemap(
    sitemap_tree: Optional[etree._Element], namespace: str
) -> List[str]:
    """
    Extracts URLs from a parsed sitemap tree.

    Args:
        sitemap_tree: The parsed XML sitemap tree
        namespace: XML namespace to use for XPath queries

    Returns:
        List of URLs extracted from the sitemap

    Raises:
        SitemapError: If there's an error extracting URLs from the sitemap
    """
    if sitemap_tree is None:
        logging.warning("Cannot extract URLs from None sitemap tree")
        return []

    try:
        # XPath query needs the namespace
        urls = sitemap_tree.xpath("//ns:loc/text()", namespaces={"ns": namespace})

        # Validate URLs
        valid_urls = []
        for url in urls:
            url = url.strip()
            if url and url.startswith(("http://", "https://")):
                valid_urls.append(url)
            else:
                logging.warning(f"Skipping invalid URL in sitemap: {url}")

        if not valid_urls and urls:
            # We found loc elements but no valid URLs
            logging.warning(
                f"Found {len(urls)} URLs in sitemap but none were valid. "
                f"Check namespace and XML structure."
            )

        return valid_urls
    except Exception as e:
        error_msg = f"Error extracting URLs from sitemap: {e}"
        logging.error(error_msg, exc_info=True)
        raise SitemapError(error_msg) from e


def scrape_page(url: str) -> Optional[Dict[str, str]]:
    """
    Scrapes the main content and title from a single page.

    Args:
        url: The URL to scrape

    Returns:
        Dictionary with url, title, and text content or None if scraping fails

    Raises:
        ScrapingError: If there's an error scraping the page
    """
    logging.info(f"Scraping: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract title
        title = (
            soup.find("title").get_text(strip=True)
            if soup.find("title")
            else "No Title Found"
        )

        # Try common content containers - might need refinement based on site structure
        content_container = _find_content_container(soup)

        if content_container:
            # Get text and clean whitespace
            text_content = content_container.get_text(separator="\n", strip=True)
            # Further clean excessive newlines
            text_content = re.sub(r"\n{3,}", "\n\n", text_content)
        else:
            logging.warning(
                f"Could not find main content container for {url}. Using body text."
            )
            # Fallback to body text if specific container not found
            text_content = (
                soup.body.get_text(separator="\n", strip=True) if soup.body else ""
            )
            text_content = re.sub(r"\n{3,}", "\n\n", text_content)

        # Validate text content
        if not text_content or len(text_content.strip()) < 50:
            logging.warning(f"Minimal or no content found at {url}")
            return None

        return {"url": url, "title": title, "text": text_content}

    except requests.exceptions.RequestException as e:
        error_msg = f"Error requesting {url}: {e}"
        logging.error(error_msg)
        # Don't raise exception here to allow continuing with other URLs
        return None
    except Exception as e:
        error_msg = f"Unexpected error scraping {url}: {e}"
        logging.error(error_msg, exc_info=True)
        # Don't raise exception here to allow continuing with other URLs
        return None


def _find_content_container(soup: BeautifulSoup) -> Optional[Any]:
    """
    Find the main content container in a BeautifulSoup object.

    Args:
        soup: BeautifulSoup object of the page

    Returns:
        The main content container element or None if not found
    """
    # Try different selectors to find main content
    main_content_selectors = [
        # Try specific tags
        soup.find("main"),
        soup.find("article"),
        # Try common classes using regex patterns
        soup.find(class_=re.compile(r"entry-content|post-content", re.I)),
        soup.find("div", class_=re.compile(r"content|post|entry|main", re.I)),
        # Try common IDs
        soup.find(id=re.compile(r"content|post|entry|main", re.I)),
    ]

    # Return the first non-None result
    for container in main_content_selectors:
        if container:
            return container

    return None


def process_sitemap_index() -> Optional[str]:
    """
    Process the sitemap index to find post sitemap URL.

    Returns:
        The post sitemap URL if found, None otherwise

    Raises:
        SitemapError: If there is an error processing the sitemap index
    """
    try:
        sitemap_index_tree = fetch_xml(SITEMAP_INDEX_URL)

        # Determine namespace from the root element
        namespace = sitemap_index_tree.nsmap.get(None)
        if not namespace:
            error_msg = "Could not determine XML namespace for sitemap index."
            logging.error(error_msg)
            raise SitemapError(error_msg)

        post_sitemap_urls = extract_urls_from_sitemap(sitemap_index_tree, namespace)
        if not post_sitemap_urls:
            error_msg = "No URLs found in sitemap index."
            logging.error(error_msg)
            raise SitemapError(error_msg)

        target_sitemap_url = None
        for url in post_sitemap_urls:
            if "post-sitemap.xml" in url:
                target_sitemap_url = url
                break

        if not target_sitemap_url:
            error_msg = "Could not find post-sitemap.xml in the index."
            logging.error(error_msg)
            raise SitemapError(error_msg)

        logging.info(f"Found post sitemap: {target_sitemap_url}")
        return target_sitemap_url

    except (XMLParsingError, ScrapingError) as e:
        # Re-raise with more context
        raise SitemapError(f"Error processing sitemap index: {e}") from e


def extract_content_urls(sitemap_url: str) -> List[str]:
    """
    Extract content URLs from a sitemap.

    Args:
        sitemap_url: URL of the sitemap to process

    Returns:
        List of content URLs found in the sitemap

    Raises:
        SitemapError: If there is an error extracting URLs from the sitemap
    """
    try:
        sitemap_tree = fetch_xml(sitemap_url)

        # Determine namespace for the sitemap
        namespace = sitemap_tree.nsmap.get(None)
        if not namespace:
            logging.warning(
                "Could not determine XML namespace for sitemap. "
                "Attempting without namespace."
            )
            # Attempt without namespace if detection failed
            try:
                content_urls = sitemap_tree.xpath("//loc/text()")
            except Exception as e:
                error_msg = f"Error extracting URLs without namespace: {e}"
                logging.error(error_msg)
                raise SitemapError(error_msg) from e
        else:
            content_urls = extract_urls_from_sitemap(sitemap_tree, namespace)

        if not content_urls:
            error_msg = f"No URLs found in sitemap: {sitemap_url}"
            logging.error(error_msg)
            raise SitemapError(error_msg)

        logging.info(f"Found {len(content_urls)} URLs in the sitemap.")
        return content_urls

    except (XMLParsingError, ScrapingError) as e:
        # Re-raise with more context
        raise SitemapError(f"Error extracting content URLs: {e}") from e


def scrape_content_urls(urls: List[str]) -> int:
    """
    Scrape content from a list of URLs and save results to OUTPUT_FILE.

    Args:
        urls: List of URLs to scrape

    Returns:
        Number of successfully scraped pages

    Raises:
        ScrapingError: If there is an error scraping the content
    """
    if not urls:
        error_msg = "No URLs provided for scraping."
        logging.error(error_msg)
        raise ScrapingError(error_msg)

    scraped_data = []
    try:
        with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as f:
            for i, url in enumerate(urls):
                data = scrape_page(url)
                if data:
                    # Write each result as a JSON line
                    json.dump(data, f, ensure_ascii=False)
                    f.write("\n")
                    scraped_data.append(data)

                # Be polite to the server
                time.sleep(REQUEST_DELAY)

                if (i + 1) % 50 == 0:
                    logging.info(f"Scraped {i + 1}/{len(urls)} pages.")

        if not scraped_data:
            logging.warning(
                f"No content was successfully scraped from {len(urls)} URLs. "
                "Check for site structure changes or access restrictions."
            )

        logging.info(f"Scraping complete. Data saved to {OUTPUT_JSONL_FILE}")
        return len(scraped_data)

    except IOError as e:
        error_msg = f"Error writing to output file {OUTPUT_JSONL_FILE}: {e}"
        logging.error(error_msg)
        raise ScrapingError(error_msg) from e
    except Exception as e:
        if isinstance(e, ScrapingError):
            # Re-raise custom exceptions
            raise
        error_msg = f"Unexpected error during scraping: {e}"
        logging.error(error_msg, exc_info=True)
        raise ScrapingError(error_msg) from e


def main() -> None:
    """
    Main function to run the scraping process.

    Steps:
    1. Log configuration settings
    2. Process the sitemap index URL
    3. Extract content URLs
    4. Scrape content from each URL
    5. Save results to a JSONL file

    Raises:
        SitemapError: If there's an error processing the sitemap
        Exception: For any unexpected errors
    """
    logging.info("Starting scrape process...")

    # Log configuration settings
    log_config()

    try:
        # Step 1: Process the sitemap index to find the post sitemap
        post_sitemap_url = process_sitemap_index()

        # Step 2: Extract content URLs from the post sitemap
        content_urls = extract_content_urls(post_sitemap_url)

        # Step 3: Scrape content from the URLs
        successful_scrapes = scrape_content_urls(content_urls)
        logging.info(f"Successfully scraped {successful_scrapes} pages.")

    except (ScrapingError, SitemapError, XMLParsingError) as e:
        # Log the error and exit with a non-zero status
        logging.error(f"Scraping pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other exceptions and provide helpful context
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
