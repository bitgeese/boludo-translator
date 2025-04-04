#!/bin/bash
# Development script to run the FastAPI app with Chainlit integrated

# Export required environment variables from .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 