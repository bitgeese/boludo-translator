"""
Argentinian Spanish Translator - Chainlit Web Application

This module provides the main Chainlit application interface for the translator.
It initializes necessary components and handles user interactions.
"""

import logging

import chainlit as cl
from langchain_core.runnables.config import RunnableConfig
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

# --- Global Initialization ---
# Declare placeholders for global objects
prompt_manager = None
vector_store = None
translation_service = None
INITIALIZATION_SUCCESSFUL = False

try:
    logger.info("Starting global initialization...")
    # 1. Initialize Prompt Manager (uses settings internally)
    prompt_manager = PromptManager()

    # 2. Load data and create vector store (uses settings internally)
    vector_store = load_vector_store_and_data()

    # 3. Initialize Translation Service (uses settings internally)
    translation_service = TranslationService(
        vector_store=vector_store, prompt_manager=prompt_manager
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

# Remove the slowapi limiter instance for messages
# message_limiter = Limiter(key_func=get_session_id) # REMOVED

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
# @message_limiter.limit("5/minute") # REMOVED Decorator
async def on_message(message: cl.Message):
    """Handle incoming text messages and provide translations."""
    # --- MANUAL Rate Limit Check (using 'limits' library directly) --- <<< CORRECTED
    session_id = get_session_id()
    # Use the limits strategy's hit() method. It returns False if the limit is exceeded.
    if not message_limit_strategy.hit(message_rate_limit, session_id):
        # Limit exceeded
        logger.warning(f"Rate limit exceeded for session {session_id}")
        await cl.ErrorMessage(
            content="Rate limit exceeded (5 messages per minute). Please wait a moment."
        ).send()
        return  # Stop processing this message
    # --- End of Rate Limit Check ---

    # Proceed with message handling only if the rate limit check passed
    try:
        # REMOVED await message_limiter.hit("5/minute", get_session_id())

        if not INITIALIZATION_SUCCESSFUL:
            await cl.ErrorMessage(content="Application not initialized.").send()
            return

        service = cl.user_session.get("translation_service")

        if not service:
            logger.error("TranslationService not found in user session.")
            await cl.ErrorMessage(
                content="Error: Translation service unavailable. "
                "Please restart the chat."
            ).send()
            return

        if not message.content:
            logger.warning("Received empty message.")
            return  # Ignore empty messages

        # Conditionally add the callback handler for step visibility
        callbacks = []
        if settings.DEBUG:
            callbacks.append(cl.LangchainCallbackHandler())
            logger.info(
                "Debug enabled: Adding LangchainCallbackHandler for step visibility."
            )

        config = RunnableConfig(callbacks=callbacks)

        # Use the service to translate, passing the config (with or without callbacks)
        if settings.DEBUG:
            # When debugging, let the callback handler manage steps
            translation_result = await service.translate_text(
                message.content, config=config
            )
        else:
            # When not debugging, show a simple progress step
            async with cl.Step(name="Translating..."):
                # Config will have empty callbacks list here
                translation_result = await service.translate_text(
                    message.content, config=config
                )
                # Optionally set step output
                # (might be redundant if result is sent immediately after)
                # step.output = translation_result

        # Send the final translation result
        await cl.Message(content=f"Translation: {translation_result}").send()

    except TranslationError as e:
        logger.error(
            f"Translation failed for '{message.content[:50]}...': {e}", exc_info=False
        )  # exc_info=False to avoid redundant stack trace from service layer
        await cl.ErrorMessage(
            content=f"Sorry, translation failed: {e}"
        ).send()  # Show specific error if safe
    except AppError as e:
        logger.error(
            f"Service error during translation for '{message.content[:50]}...': {e}",
            exc_info=True,
        )
        await cl.ErrorMessage(
            content="Sorry, an application error occurred during translation."
        ).send()
    except Exception as e:  # Catch other potential exceptions from the core logic
        # This generic catch might now be redundant if specific errors are handled
        # but kept for safety, ensuring RateLimitExceeded is handled first.
        logger.error(
            f"Unexpected error during translation for '{message.content[:50]}...': {e}",
            exc_info=True,
        )
        await cl.ErrorMessage(
            content=(
                "Sorry, an unexpected error occurred during translation. "
                "Please try again."
            )
        ).send()


# Removed @cl.on_settings_update as it wasn't used after refactor
# Removed @cl.on_chat_end/@cl.on_stop as they were empty
