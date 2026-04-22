"""config.py - Load and manage application configuration."""

import json
import os
from pathlib import Path
from typing import Any


CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config() -> dict[str, Any]:
    """Load configuration from config.json."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}\n"
            "Please ensure config.json exists in the project root."
        )
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_chroma_dir() -> str:
    """Get the ChromaDB directory path (absolute)."""
    config = load_config()
    chroma_dir = config["chroma"]["directory"]
    if not os.path.isabs(chroma_dir):
        chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), chroma_dir)
    return chroma_dir


def get_ragignore_path() -> str:
    """Get the .ragignore file path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ragignore")


def get_raginclude_path() -> str:
    """Get the .raginclude file path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".raginclude")
