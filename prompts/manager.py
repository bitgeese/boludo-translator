"""
Prompt Manager for Argentinian Spanish Translator

This module handles loading and managing prompts from markdown files,
providing a centralized way to manage and update prompts.
"""

import os
from pathlib import Path


class PromptManager:
    """Manages loading and accessing prompts from markdown files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt markdown files
        """
        self.prompts_dir = Path(prompts_dir)
        
    def load_prompt(self, filename: str) -> str:
        """
        Load a prompt from a markdown file.
        
        Args:
            filename: Name of the markdown file (without extension)
            
        Returns:
            The prompt content as a string
        """
        file_path = self.prompts_dir / f"{filename}.md"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
            
    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self.load_prompt("system")
        
    @property
    def translation_prompt(self) -> str:
        """Get the translation prompt template."""
        return self.load_prompt("translation") 