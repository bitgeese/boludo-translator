"""
Argentinian Spanish Translator - Chainlit Web Application

This module provides the main Chainlit application interface for the translator.
It initializes necessary components and handles user interactions.
"""

import logging
import os

import chainlit as cl
import psutil  # For memory tracking
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from config import settings
from core.data_loader import load_vector_store_and_data

# Import custom exceptions
from core.exceptions import AppError, DataLoaderError, PromptError, TranslationError
from core.prompt_manager import PromptManager
from services.translation_service import TranslationService

# --- Logging Setup (as early as possible) ---
# Configure root logger based on settings
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# You might want to suppress overly verbose library logs here, e.g.:
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Memory Management Constants ---
# Maximum number of message pairs (user+assistant) to keep in memory
MAX_HISTORY_LENGTH = settings.MAX_HISTORY_LENGTH


# Function to trim message history to prevent memory bloat
def trim_message_history(session_id: str) -> None:
    """
    Trims the message history for a session when it gets too long.
    This helps prevent memory buildup in long conversations.

    Args:
        session_id: The ID of the current session
    """
    try:
        # Get current message history
        history = cl.user_session.get("message_history", [])

        # If history exceeds max length, trim it
        if len(history) > MAX_HISTORY_LENGTH * 2:  # Each exchange has 2 messages
            # Keep only the most recent messages
            history = history[-MAX_HISTORY_LENGTH * 2 :]
            cl.user_session.set("message_history", history)
            logger.info(
                f"Trimmed message history for session {session_id} "
                f"to {len(history)} messages"
            )
    except Exception as e:
        # Log but don't crash if history trimming fails
        logger.warning(f"Failed to trim message history: {e}")


# --- Global Initialization ---
# Declare placeholders for global objects
prompt_manager = None
vector_index = None
translation_service = None
INITIALIZATION_SUCCESSFUL = False

try:
    logger.info("Starting global initialization...")
    # 1. Initialize Prompt Manager (uses settings internally)
    prompt_manager = PromptManager()

    # 2. Load data and create vector index with a limited dataset for debugging
    # Use a small number like 20 to avoid hitting rate limits during testing
    vector_index = load_vector_store_and_data(debug_limit=20)

    # 3. Initialize Translation Service (uses settings internally)
    translation_service = TranslationService(
        vector_index=vector_index, prompt_manager=prompt_manager
    )
    logger.info("Global initialization complete.")
    INITIALIZATION_SUCCESSFUL = True

# Catch specific initialization errors
except (DataLoaderError, PromptError, AppError) as e:
    logger.critical(
        f"Fatal error during application initialization: {e}", exc_info=True
    )
    INITIALIZATION_SUCCESSFUL = False
except Exception as e:
    # Catch any other unexpected exceptions during init
    logger.critical(
        f"An unexpected fatal error occurred during application initialization: {e}",
        exc_info=True,
    )
    INITIALIZATION_SUCCESSFUL = False


# --- Rate Limiter for Messages (using 'limits' directly) ---
# Key function uses the Chainlit session ID
def get_session_id():
    try:
        return cl.context.session.id
    except Exception:
        # Fallback if context is not available (should not happen in on_message)
        logger.warning("Chainlit context/session ID not found for rate limiter key.")
        return "unknown_session"


# Configure limits library directly for message limiting
message_limit_storage = MemoryStorage()
message_limit_strategy = FixedWindowRateLimiter(message_limit_storage)
message_rate_limit = parse("5/minute")  # Use limits.parse


# --- Helper Functions for Message Processing ---
async def check_initialization() -> bool:
    """Check if the application is properly initialized."""
    if not INITIALIZATION_SUCCESSFUL:
        await cl.ErrorMessage(content="Application not initialized.").send()
        return False
    return True


async def get_translation_service():
    """Get the translation service from the user session."""
    service = cl.user_session.get("translation_service")
    if not service:
        logger.error("TranslationService not found in user session.")
        await cl.ErrorMessage(
            content="Error: Translation service unavailable. "
            "Please restart the chat."
        ).send()
        return None
    return service


# Custom callback handler for Chainlit integration with LlamaIndex
class ChainlitCallbackHandler:
    """Custom callback handler for LlamaIndex that integrates with Chainlit steps."""

    async def on_retrieval_start(self, query):
        step = cl.Step(name="Retrieving context")
        await step.send()
        self.current_step = step

    async def on_retrieval_end(self, nodes):
        if hasattr(self, "current_step"):
            await self.current_step.end()

    async def on_llm_start(self, prompt):
        step = cl.Step(name="Generating translation")
        await step.send()
        self.current_step = step

    async def on_llm_end(self, response):
        if hasattr(self, "current_step"):
            await self.current_step.end()


async def perform_translation(service, message_content, callback_handler=None):
    """Perform the actual translation using the service."""
    if settings.DEBUG and callback_handler:
        # In debug mode, just log the callback handler but don't use it
        logger.info("Debug mode: callback handler is enabled but not used")

    # Show a simple progress step
    async with cl.Step(name="Translating..."):
        return await service.translate_text(message_content)


async def log_memory_usage(session_id):
    """Log current memory usage for monitoring."""
    try:
        # Use psutil to get memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        logger.info(f"Memory usage: {memory_mb:.2f} MB for session {session_id}")

        # If memory usage is high, log a warning
        if memory_mb > 400:  # 400MB is getting close to the 512MB limit
            logger.warning(f"High memory usage detected: {memory_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to log memory usage: {e}")


# --- Chainlit Event Handlers ---


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    if not INITIALIZATION_SUCCESSFUL:
        await cl.ErrorMessage(
            content="Application failed to initialize. Please check the logs."
        ).send()
        return

    # Store the globally initialized service in the user session
    # This is generally safer than relying on global variables in async contexts
    cl.user_session.set("translation_service", translation_service)

    # Send welcome message
    await cl.Message(
        content=(
            "¡Bienvenido che! I'm your Argentinian Spanish translator. "
            "Send me a message in English or Spanish, and I'll translate it "
            "to casual Argentinian Spanish."
        )
    ).send()
    logger.info("Chat session started.")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming text messages and provide translations."""
    # --- Rate Limit Check ---
    session_id = get_session_id()
    if not message_limit_strategy.hit(message_rate_limit, session_id):
        # Limit exceeded
        logger.warning(f"Rate limit exceeded for session {session_id}")
        await cl.ErrorMessage(
            content="Rate limit exceeded (5 messages per minute). Please wait a moment."
        ).send()
        return

    # --- Memory Management ---
    trim_message_history(session_id)

    # Track this message in history
    history = cl.user_session.get("message_history", [])
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", history)

    try:
        # Basic validations
        if not await check_initialization():
            return

        service = await get_translation_service()
        if not service:
            return

        if not message.content:
            logger.warning("Received empty message.")
            return

        # Setup for translation
        callback_handler = None
        if settings.DEBUG:
            callback_handler = ChainlitCallbackHandler()
            logger.info("Debug enabled: Added ChainlitCallbackHandler.")

        # Perform translation
        translation_result = await perform_translation(
            service, message.content, callback_handler
        )

        # Send result
        await cl.Message(content=f"Translation: {translation_result}").send()

        # Update history
        history = cl.user_session.get("message_history", [])
        history.append(
            {"role": "assistant", "content": f"Translation: {translation_result}"}
        )
        cl.user_session.set("message_history", history)

    except TranslationError as e:
        logger.error(
            f"Translation failed for '{message.content[:50]}...': {e}", exc_info=False
        )
        await cl.ErrorMessage(content=f"Sorry, translation failed: {e}").send()
    except AppError as e:
        logger.error(
            f"Service error during translation for '{message.content[:50]}...': {e}",
            exc_info=True,
        )
        await cl.ErrorMessage(
            content="Sorry, an application error occurred during translation."
        ).send()
    except Exception as e:
        logger.error(
            f"Unexpected error during translation for '{message.content[:50]}...': {e}",
            exc_info=True,
        )
        await cl.ErrorMessage(
            content="Sorry, an unexpected error occurred during translation."
        ).send()
    finally:
        # Log memory usage
        await log_memory_usage(session_id)


@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when a chat session ends."""
    try:
        # Get the session ID for logging
        session_id = (
            cl.context.session.id if hasattr(cl.context, "session") else "unknown"
        )
        logger.info(f"Cleaning up resources for ending session {session_id}")

        # Instead of clear(), reset specific keys we care about
        if hasattr(cl, "user_session"):
            # Get all keys in the session
            keys = list(cl.user_session.keys())
            # Remove each key individually
            for key in keys:
                cl.user_session.pop(key, None)

            logger.info(f"Successfully cleaned up resources for session {session_id}")
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}", exc_info=True)
