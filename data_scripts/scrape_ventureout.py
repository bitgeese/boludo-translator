import json
import logging
import re
import time

import requests
from bs4 import BeautifulSoup
from lxml import etree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SITEMAP_INDEX_URL = "https://ventureoutspanish.com/sitemap_index.xml"
OUTPUT_FILE = "ventureout_data.jsonl"
REQUEST_DELAY = 1  # Delay in seconds between requests to be polite

# Standard browser headers to avoid 406 errors
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


def fetch_xml(url):
    """Fetches and parses XML content from a URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        # Use lxml for robust XML parsing
        parser = etree.XMLParser(recover=True)
        tree = etree.fromstring(response.content, parser=parser)
        return tree
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching XML {url}: {e}")
        return None
    except etree.XMLSyntaxError as e:
        logging.error(f"Error parsing XML {url}: {e}")
        return None


def extract_urls_from_sitemap(sitemap_tree, namespace):
    """Extracts URLs from a parsed sitemap tree."""
    urls = []
    if sitemap_tree is not None:
        # XPath query needs the namespace
        urls = sitemap_tree.xpath("//ns:loc/text()", namespaces={"ns": namespace})
    return urls


def scrape_page(url):
    """Scrapes the main content and title from a single page."""
    logging.info(f"Scraping: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        title = (
            soup.find("title").get_text(strip=True)
            if soup.find("title")
            else "No Title Found"
        )

        # Try common content containers - might need refinement based on site structure
        content_container = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=re.compile(r"entry-content|post-content", re.I))
            or soup.find("div", class_=re.compile(r"content|post|entry|main", re.I))
        )

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

        return {"url": url, "title": title, "text": text_content}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error scraping {url}: {e}")
        return None


def process_sitemap_index():
    """
    Process the sitemap index to find post sitemap URL.
    Returns the post sitemap URL if found, None otherwise.
    """
    sitemap_index_tree = fetch_xml(SITEMAP_INDEX_URL)

    if not sitemap_index_tree:
        logging.error("Could not fetch sitemap index. Exiting.")
        return None

    # Determine namespace from the root element
    namespace = sitemap_index_tree.nsmap.get(None)
    if not namespace:
        logging.error("Could not determine XML namespace for sitemap index. Exiting.")
        return None

    post_sitemap_urls = extract_urls_from_sitemap(sitemap_index_tree, namespace)
    target_sitemap_url = None
    for url in post_sitemap_urls:
        if "post-sitemap.xml" in url:
            target_sitemap_url = url
            break

    if not target_sitemap_url:
        logging.error("Could not find post-sitemap.xml in the index. Exiting.")
        return None

    logging.info(f"Found post sitemap: {target_sitemap_url}")
    return target_sitemap_url


def extract_content_urls(sitemap_url):
    """
    Extract content URLs from a sitemap.
    Returns a list of URLs if successful, empty list otherwise.
    """
    sitemap_tree = fetch_xml(sitemap_url)

    if not sitemap_tree:
        logging.error("Could not fetch sitemap. Exiting.")
        return []

    # Determine namespace for the sitemap
    namespace = sitemap_tree.nsmap.get(None)
    if not namespace:
        logging.warning(
            "Could not determine XML namespace for sitemap. "
            "Attempting without namespace."
        )
        # Attempt without namespace if detection failed
        content_urls = sitemap_tree.xpath("//loc/text()")
    else:
        content_urls = extract_urls_from_sitemap(sitemap_tree, namespace)

    logging.info(f"Found {len(content_urls)} URLs in the sitemap.")
    return content_urls


def scrape_content_urls(urls):
    """
    Scrape content from a list of URLs and save results to OUTPUT_FILE.
    Returns the number of successfully scraped pages.
    """
    scraped_data = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
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

    logging.info(f"Scraping complete. Data saved to {OUTPUT_FILE}")
    return len(scraped_data)


def main():
    """Main execution function with reduced complexity."""
    logging.info("Starting scrape process...")

    # Step 1: Process the sitemap index to find the post sitemap
    post_sitemap_url = process_sitemap_index()
    if not post_sitemap_url:
        return

    # Step 2: Extract content URLs from the post sitemap
    content_urls = extract_content_urls(post_sitemap_url)
    if not content_urls:
        return

    # Step 3: Scrape content from the URLs
    successful_scrapes = scrape_content_urls(content_urls)
    logging.info(f"Successfully scraped {successful_scrapes} pages.")


if __name__ == "__main__":
    main()
