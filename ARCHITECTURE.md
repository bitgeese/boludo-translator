# Architecture Documentation: Argentine Spanish Learning RAG System

This document provides a comprehensive overview of the Argentine Spanish Learning RAG (Retrieval Augmented Generation) system architecture, explaining components, data flow, and integration points.

## System Overview

The system combines multiple data sources about Argentine Spanish, processes and embeds this information, and provides a conversational interface through a Chainlit web application. The architecture follows a RAG pattern where user queries are used to retrieve relevant context from a vector database before generating responses with an LLM.

## Key Components

### 1. Data Processing Pipeline

The data processing pipeline is responsible for acquiring, cleaning, and preparing data for the vector database.

#### Data Acquisition
- **scrape_ventureout.py**: Scrapes articles from VentureOut Spanish blog
  - Processes sitemaps to find content URLs
  - Extracts clean content from web pages
  - Outputs data in JSONL format with metadata

#### Data Embedding
- **embed_data.py**: Processes and embeds data from multiple sources
  - Loads data from JSONL and CSV files
  - Cleans and normalizes text
  - Splits documents into chunks with overlap
  - Creates vector embeddings using OpenAI's embedding model
  - Stores embeddings in a vector database (FAISS or Chroma)

### 2. Core Components

The core directory contains modules that provide functionality used both in the data pipeline and the web application.

- **text_utils.py**: Centralized text processing utilities
  - Provides text cleaning functions to normalize content
  - Used by both the data processing pipeline and loaders

- **data_loader.py**: Handles loading and retrieval of vector data
  - Manages vector store initialization
  - Provides unified access to both CSV and web-scraped data
  - Creates and manages the embedding functions

- **ventureout_loader.py**: Specialized loader for VentureOut data
  - Processes VentureOut-specific data format
  - Converts raw data to Langchain documents

- **exceptions.py**: Custom exception classes
  - Provides hierarchical exception structure
  - Enables consistent error handling across the application

### 3. Configuration

- **config.py**: Centralized configuration system
  - Defines default values for all configurable parameters
  - Allows overriding values via environment variables
  - Provides logging of current configuration
  - Manages file paths, chunking parameters, and other settings

### 4. Web Application

- **app.py**: Chainlit web application
  - Provides conversational interface
  - Integrates with the vector store for knowledge retrieval
  - Constructs prompts with retrieved context
  - Manages user sessions and conversations

## Data Flow

1. **Data Acquisition & Processing**:
   ```
   Web Content → scrape_ventureout.py → JSONL File
   CSV Data    → (manual creation) → CSV File
   ```

2. **Embedding & Storage**:
   ```
   JSONL File → embed_data.py → Vector Store
   CSV File   → embed_data.py → Vector Store
   ```

3. **Query & Response**:
   ```
   User Query → Chainlit → Vector Search → Context Retrieval → LLM → Response
   ```

## Integration Points

### OpenAI API Integration
- The system relies on OpenAI's API for two primary functions:
  - Creating embeddings of text chunks for vector storage
  - Generating responses with LLM using retrieved context

### Chainlit Integration
- The Chainlit framework is used to create the conversational interface
- Integration happens in the app.py file, where:
  - The vector store is initialized on application startup
  - Messages are processed to retrieve relevant context
  - Responses are generated and displayed to the user

## Environment Variables

The system can be configured through the following environment variables:

- `OPENAI_API_KEY`: Required for embeddings and LLM queries
- `CHROMA_PERSIST_DIRECTORY`: Path to store the vector database
- `CSV_INPUT_FILE`: Path to the CSV data file
- `JSONL_INPUT_FILE`: Path to the JSONL data file
- `OUTPUT_JSONL_FILE`: Output location for scraped content
- `REQUEST_DELAY`: Time to wait between scraping requests
- And many more defined in config.py

## Scaling Considerations

The current architecture can be scaled in several ways:

1. **Data Sources**: Additional scrapers can be created for other Spanish learning resources
2. **Vector Stores**: The system can be extended to use different vector stores
3. **Embedding Models**: Custom embedding models could be integrated
4. **Deployment**: The application can be containerized for cloud deployment

## Future Enhancements

Potential improvements to the architecture include:

1. **Streaming Responses**: Implementing streaming for a better user experience
2. **User Feedback Loop**: Capturing feedback to improve retrieval quality
3. **Hybrid Search**: Combining vector search with keyword search
4. **Progressive Data Updates**: Implementing incremental updates to the vector store
5. **Custom Embedding Model**: Training a domain-specific embedding model

## Dependencies

Major dependencies include:
- LangChain for RAG implementation
- FAISS or Chroma for vector storage
- BeautifulSoup for web scraping
- OpenAI API for embeddings and generation
- Pandas for CSV data processing
- Chainlit for the web interface 