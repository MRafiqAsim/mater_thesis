"""
Central prompt loader — reads all LLM prompts from config/prompts.json.

The JSON is organized by pipeline layer → prompt section → key:

    {
      "silver": {
        "pii_detection": {
          "system_prompt": "...",
          "user_prompt": "...",
          "temperature": 0.0
        }
      }
    }

Usage:
    from prompt_loader import get_prompt

    system = get_prompt("silver", "pii_detection", "system_prompt")
    temp   = get_prompt("silver", "pii_detection", "temperature")
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE: Optional[dict] = None


def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    prompts_path = Path(__file__).resolve().parent.parent / "config" / "prompts.json"
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
            logger.info(f"Loaded prompts from {prompts_path}")
    else:
        logger.warning(f"Prompts config not found at {prompts_path}, using empty dict")
        _CACHE = {}

    return _CACHE


def get_prompt(layer: str, section: str, key: str, default: Any = None) -> Any:
    """
    Get a prompt value from config/prompts.json.

    Args:
        layer: Pipeline layer ("bronze", "silver", "gold", "retrieval")
        section: Prompt group (e.g. "pii_detection", "generation")
        key: Key within the group (e.g. "system_prompt", "temperature")
        default: Fallback if not found

    Returns:
        The prompt string, number, or list
    """
    data = _load()
    return data.get(layer, {}).get(section, {}).get(key, default)


def format_prompt(template: str, **kwargs) -> str:
    """
    Safely format a prompt template, preserving literal braces.

    Replaces only {key} placeholders that match kwargs keys.
    All other braces (including JSON examples) are left untouched.
    """
    import re

    # First, escape ALL braces
    safe = template.replace("{", "{{").replace("}", "}}")

    # Then un-escape only the known kwargs placeholders
    for key in kwargs:
        safe = safe.replace("{{" + key + "}}", "{" + key + "}")

    return safe.format(**kwargs)


def get_section(layer: str, section: str) -> dict:
    """Get an entire prompt section."""
    data = _load()
    return data.get(layer, {}).get(section, {})
