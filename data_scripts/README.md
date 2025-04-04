# Data Scripts for Argentinian Spanish Translator

This directory contains scripts for collecting, processing, and preparing data for the Argentinian Spanish Translator application.

## Available Scripts

### VentureOut Spanish Scraper
**File:** `scrape_ventureout.py`

This script scrapes content from [Venture Out Spanish](https://ventureoutspanish.com/), a website dedicated to Rioplatense Spanish from Argentina. The script:
- Fetches the sitemap
- Extracts URLs for blog posts and articles
- Scrapes the content of each page
- Saves the data to a JSONL file (`ventureout_data.jsonl`)

**Usage:**
```bash
python scrape_ventureout.py
```

### Data Cleaning and Testing
**Files:** 
- `test_cleaning.py`
- `embed_data.py`

These scripts demonstrate the text cleaning process that removes boilerplate content from scraped pages.

**Usage:**
```bash
python test_cleaning.py  # Tests cleaning on sample documents
```

## Integration with Chainlit App

The main application (`app.py` in the root directory) integrates the VentureOut data automatically. The integration:

1. The data loader in `core/data_loader.py` loads and combines multiple data sources:
   - Phrases from `data/phrases.csv`
   - VentureOut content from `data/ventureout_data.jsonl`

2. The vector store uses both data sources to provide context for translations.

3. The translation prompts have been enhanced to use both:
   - Specific phrases for direct translations
   - Article content for contextual understanding

## Adding More Data

To update or refresh the VentureOut data:

1. Run the scraping script in this directory:
   ```bash
   python scrape_ventureout.py
   ```

2. Copy the resulting JSONL file to the `data` directory:
   ```bash
   cp ventureout_data.jsonl ../data/
   ```

3. Restart the Chainlit application to load the new data.

## Future Improvements

Consider adding more data sources specialized in Argentinian Spanish, such as:
- Argentine news sites
- Local forums or social media content
- Additional specialized blogs

For each new source, create a dedicated scraper script and update the data loader to incorporate it. 