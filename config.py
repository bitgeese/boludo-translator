import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- Data Configuration ---
PHRASES_CSV_PATH = "data/phrases.csv"

# --- Prompt Configuration ---
PROMPTS_DIR = "prompts"
SYSTEM_PROMPT_FILE = "system.md"
TRANSLATION_PROMPT_FILE = "translation.md"

# --- Embedding/Vector Store Configuration ---
# (Add any relevant configurations here if needed later, e.g., embedding model name)

# --- Translator Configuration ---
TRANSLATOR_MODEL_NAME = "gpt-4o-mini" # Example, adjust as needed 