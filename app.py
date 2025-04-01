"""
Argentinian Spanish Translator - Chainlit Web Application

This module provides a web interface for translating text to Argentinian Spanish.
"""

import asyncio
from dotenv import load_dotenv
import chainlit as cl
from chainlit.logger import logger

# Import custom modules
from loaders import ArgentinianSpanishLoader
from translator import ArgentinianTranslator
from prompts.manager import PromptManager

# Load environment variables
load_dotenv()


# Chainlit event handlers
@cl.on_settings_update
async def setup_agent(settings):
    """Update user settings when changed from the UI."""
    cl.user_session.set("settings", settings)


@cl.on_chat_start
async def start():
    """Initialize chat session with the translator."""
    # Configure initial settings
    settings = {
        "chat_profile": {
            "name": "Argentinian Spanish Translator",
            "description": "Translate text to casual Argentinian Spanish"
        }
    }
    cl.user_session.set("settings", settings)
    
    # Load prompts
    prompt_manager = PromptManager()
    
    # Send welcome message
    await cl.Message(
        content="¡Bienvenido che! I'm your Argentinian Spanish translator. Send me a message in English or Spanish, and I'll translate it to casual Argentinian Spanish."
    ).send()
    
    # Load Argentinian Spanish data
    loader = ArgentinianSpanishLoader("phrases.csv")
    vector_store, reference_df = loader.load_and_process()
    
    # Initialize translator
    translator = ArgentinianTranslator(vector_store, reference_df)
    
    # Store in user session
    cl.user_session.set("translator", translator)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming text messages and provide translations."""
    translator = cl.user_session.get("translator")
    
    if translator:
        # Translate the message
        translation_result = await translator.translate(message.content)
        
        # Send the translation result
        await cl.Message(
            content=f"Translation: {translation_result['argentinian_translation']}"
        ).send()
    else:
        await cl.Message(
            content="Error: Translator not initialized. Please try again."
        ).send()


@cl.on_chat_end
@cl.on_stop
async def on_end():
    """Clean up resources when chat session terminates."""
    pass