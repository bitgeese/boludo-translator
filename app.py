"""
Argentinian Spanish Translator - Chainlit Web Application

This module provides a web interface for translating text to Argentinian Spanish,
with support for both text and voice interactions via OpenAI's Realtime API.
"""

# import os
import asyncio
from dotenv import load_dotenv
# from openai import AsyncOpenAI

import chainlit as cl
from uuid import uuid4
from chainlit.logger import logger

from realtime import RealtimeClient
from realtime.tools import tools

# client = AsyncOpenAI()

load_dotenv()

# Configure Chainlit settings
@cl.on_settings_update
async def setup_agent(settings):
    """Update user settings when changed from the UI."""
    cl.user_session.set("settings", settings)

@cl.on_chat_start
async def start():
    """Initialize chat session with the translator and realtime capabilities."""
    # Configure initial settings
    settings = {
        "voice_enabled": False,
        "chat_profile": {
            "name": "Argentinian Spanish Translator",
            "description": "Translate text to casual Argentinian Spanish"
        }
    }
    cl.user_session.set("settings", settings)
    
    await cl.Message(
        content="¡Bienvenido che! I'm your Argentinian Spanish translator. Send me a message in English or Spanish, and I'll translate it to casual Argentinian Spanish. You can also use voice mode by pressing 'P'."
    ).send()
    
    # Load Argentinian Spanish data
    loader = ArgentinianSpanishLoader("phrases.csv")
    vector_store, reference_df = loader.load_and_process()
    
    # Initialize translator
    translator = ArgentinianTranslator(vector_store, reference_df)
    
    # Store in user session
    cl.user_session.set("translator", translator)
    
    await setup_openai_realtime()

# @cl.on_chat_start
# async def start():
#     await cl.Message(
#         content="Welcome to the Chainlit x OpenAI realtime example. Press `P` to talk!"
#     ).send()
#     await setup_openai_realtime()


# @cl.on_message
# async def on_message(message: cl.Message):
#     openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
#     if openai_realtime and openai_realtime.is_connected():
#         # TODO: Try image processing with message.elements
#         await openai_realtime.send_user_message_content(
#             [{"type": "input_text", "text": message.content}]
#         )
#     else:
#         await cl.Message(
#             content="Please activate voice mode before sending messages!"
#         ).send()


@cl.on_audio_start
async def on_audio_start():
    """Initialize audio connection when user activates voice mode."""
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(
            content=f"Failed to connect to OpenAI realtime: {e}"
        ).send()
        return False


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Process incoming audio chunks when user is speaking."""
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime.is_connected():
        await openai_realtime.append_input_audio(chunk.data)
    else:
        logger.info("RealtimeClient is not connected")


@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    """Clean up resources when audio input ends or chat session terminates."""
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()


# Add to your imports
from loaders import ArgentinianSpanishLoader
from translator import ArgentinianTranslator

# Modify your @cl.on_message function
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming text messages and provide translations."""
    translator = cl.user_session.get("translator")
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    
    if translator:
        # Translate the message
        translation_result = await translator.translate(message.content)
        
        # Send the translation result
        await cl.Message(
            content=f"Translation: {translation_result['argentinian_translation']}"
        ).send()
        
        # If realtime client is connected and voice mode is active, also send to it
        if openai_realtime and openai_realtime.is_connected():
            await openai_realtime.send_user_message_content(
                [{"type": "input_text", "text": translation_result["argentinian_translation"]}]
            )
    else:
        await cl.Message(
            content="Error: Translator not initialized. Please try again."
        ).send()

async def setup_openai_realtime():
    """
    Initialize and configure the OpenAI Realtime Client for voice interaction.
    Sets up event handlers for audio processing and transcription.
    """
    openai_realtime = RealtimeClient()
    cl.user_session.set("track_id", str(uuid4()))

    async def handle_conversation_updated(event):
        """Handle updates to the conversation, particularly audio events."""
        delta = event.get("delta")
        if delta:
            # Process different types of deltas
            if "audio" in delta:
                audio = delta["audio"]  # Int16Array, audio added
                await cl.context.emitter.send_audio_chunk(
                    cl.OutputAudioChunk(
                        mimeType="pcm16",
                        data=audio,
                        track=cl.user_session.get("track_id"),
                    )
                )
            if "transcript" in delta:
                # Handle transcript updates if needed
                pass
            if "arguments" in delta:
                # Handle function argument updates if needed
                pass

    async def handle_item_completed(item):
        """Process completed conversation items."""
        # Additional processing can be added here if needed
        pass

    async def handle_conversation_interrupt(event):
        """Handle interruptions in the conversation flow."""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()

    async def handle_error(event):
        """Log errors from the realtime client."""
        logger.error(event)

    # Register event handlers
    openai_realtime.on("conversation.updated", handle_conversation_updated)
    openai_realtime.on("conversation.item.completed", handle_item_completed)
    openai_realtime.on("conversation.interrupted", handle_conversation_interrupt)
    openai_realtime.on("error", handle_error)

    # Store client in session and register tools
    cl.user_session.set("openai_realtime", openai_realtime)
    coros = [
        openai_realtime.add_tool(tool_def, tool_handler)
        for tool_def, tool_handler in tools
    ]
    await asyncio.gather(*coros)