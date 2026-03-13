"""
Email Text Cleaner — semantic-safe preprocessing.

Normalizes whitespace, removes control characters and email artifacts
without altering semantic meaning. Called in Silver before chunking.
"""

import re
import html as html_module

from .disclaimer_remover import remove_disclaimers


def clean_email_text(text: str) -> str:
    """Clean email body text while preserving semantic content."""
    if not text:
        return text

    # 0. Remove legal disclaimers first (confidentiality notices, virus warnings)
    text = remove_disclaimers(text)

    # 1. Normalize line endings: \r\n and \r → \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Decode HTML entities that survived extraction (&amp; &lt; &gt; &nbsp; &#160; etc.)
    text = html_module.unescape(text)

    # 3. Strip control characters (keep \n and \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 4. Remove zero-width / invisible Unicode characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]', '', text)

    # 5. Normalize Unicode whitespace (non-breaking space, etc.) → regular space
    text = re.sub(r'[\u00a0\u2000-\u200a\u202f\u205f\u3000]', ' ', text)

    # 6. Remove tracking pixel remnants and image tags that leaked through
    text = re.sub(
        r'\[?(?:image|img|cid|tracking)[:\s]*[^\]\n]*\]?',
        '', text, flags=re.IGNORECASE
    )

    # 7. Remove MIME boundary lines (------=_Part_12345)
    text = re.sub(r'^-{2,}=_\S+.*$', '', text, flags=re.MULTILINE)

    # 8. Clean forwarded/reply separator lines — normalize to standard form
    #    Handles: -----Original Message-----,  ___ (Outlook underscores),
    #    -----Forwarded message-----
    #    Keep the line but normalize to a clean marker
    text = re.sub(
        r'^[-_]{3,}\s*(?:Original Message|Forwarded message|Oorspronkelijk bericht|Doorgestuurd bericht)\s*[-_]{3,}\s*$',
        '--- Forwarded ---',
        text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 9. Normalize quoted-reply markers: collapse deep nesting (>>> → >) but KEEP content
    #    "> > > text" → "> text"
    text = re.sub(r'^(?:>\s*){2,}', '> ', text, flags=re.MULTILINE)

    # 10. Collapse repeated spaces within lines (preserve leading indent)
    text = re.sub(r'(?<=\S) {2,}', ' ', text)

    # 11. Strip trailing whitespace on each line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # 12. Collapse 3+ consecutive blank lines → 2 (preserves paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 13. Strip leading/trailing whitespace
    text = text.strip()

    return text
