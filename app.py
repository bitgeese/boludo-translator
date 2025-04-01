"""
Argentinian Spanish Translator - Chainlit Web Application

This module provides the main Chainlit application interface for the translator.
It initializes necessary components and handles user interactions.
"""

import logging
import chainlit as cl
from chainlit.logger import logger as chainlit_logger # Alias to avoid conflict

# Import core components and service
from core.data_loader import load_vector_store_and_data
from core.prompt_manager import PromptManager
from services.translation_service import TranslationService
import config # Import config to ensure it's loaded (e.g., environment variables)

# --- Global Initialization ---
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("Starting global initialization...")
    # 1. Initialize Prompt Manager (loads prompts)
    prompt_manager = PromptManager()

    # 2. Load data and create vector store
    vector_store, reference_df = load_vector_store_and_data()
    # Note: reference_df is currently loaded but not used after refactoring.
    # Consider removing it if the translator no longer needs it directly.
    
    # 3. Initialize Translation Service (creates translator instance)
    translation_service = TranslationService(
        vector_store=vector_store, 
        prompt_manager=prompt_manager
    )
    logger.info("Global initialization complete.")
    
    # Flag to indicate successful initialization
    INITIALIZATION_SUCCESSFUL = True
    
except Exception as e:
    logger.exception("Fatal error during application initialization.")
    INITIALIZATION_SUCCESSFUL = False
    # You might want to prevent the app from starting or show a critical error
    # For now, we'll let Chainlit start but handlers will check the flag.

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
        content="¡Bienvenido che! I'm your Argentinian Spanish translator. Send me a message in English or Spanish, and I'll translate it to casual Argentinian Spanish."
    ).send()
    logger.info("Chat session started.")

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming text messages and provide translations."""
    if not INITIALIZATION_SUCCESSFUL:
         await cl.ErrorMessage(content="Application not initialized.").send()
         return
         
    service = cl.user_session.get("translation_service")
    
    if not service:
        logger.error("TranslationService not found in user session.")
        await cl.ErrorMessage(content="Error: Translation service unavailable. Please restart the chat.").send()
        return
        
    if not message.content:
        logger.warning("Received empty message.")
        return # Ignore empty messages

    try:
        # Use the service to translate
        async with cl.Step(name="Translate"):
            translation_result = await service.translate_text(message.content)
            
            # Send the translation result
            await cl.Message(
                content=f"Translation: {translation_result}"
            ).send()
            
    except Exception as e:
        logger.error(f"Error during translation for message '{message.content[:50]}...': {e}", exc_info=True)
        await cl.ErrorMessage(content="Sorry, I encountered an error during translation. Please try again.").send()

# Removed @cl.on_settings_update as it wasn't used after refactor
# Removed @cl.on_chat_end/@cl.on_stop as they were empty