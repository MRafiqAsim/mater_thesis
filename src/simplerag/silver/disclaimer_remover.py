"""
Email Disclaimer Remover

Removes common email signatures, disclaimers, and boilerplate text
before processing to reduce noise and token usage.
"""

import re
from typing import List, Tuple


class DisclaimerRemover:
    """
    Removes email disclaimers, signatures, and boilerplate.

    Common patterns:
    - Legal disclaimers ("This message is intended for...")
    - Confidentiality notices
    - Email signatures with contact info
    - Auto-generated footers
    """

    # Phrases that START a disclaimer block (case-insensitive)
    DISCLAIMER_STARTERS = [
        # English
        r"important notice:",
        r"confidentiality notice:",
        r"disclaimer:",
        r"legal disclaimer:",
        r"this (email|e-mail|message|communication|electronic mail) (is|was) intended",
        r"this (email|e-mail|message) (and any|contains|may contain)",
        r"the information (contained|in this)",
        r"if you (are not|have received) the intended",
        r"if you received this (message|email|e-mail) in error",
        r"please (note|be advised|consider the environment)",
        r"privileged (and )?confidential",
        r"confidential and (may be )?privileged",
        r"strictly (confidential|prohibited)",
        r"any (unauthorized|unlawful) (use|disclosure|distribution|copying)",
        r"dissemination, distribution,? (or|and) copying",
        r"do not (copy|forward|distribute|print)",
        r"think before (you )?print",
        r"consider the environment before printing",
        r"save (paper|trees)",
        r"go green",

        # Auto-generated
        r"sent from my (iphone|ipad|android|blackberry|mobile|samsung)",
        r"get outlook for",
        r"powered by",

        # Unsubscribe
        r"to unsubscribe",
        r"click here to unsubscribe",
        r"manage (your )?(email )?preferences",
        r"update your preferences",
    ]

    # Phrases that indicate END of useful content
    CONTENT_ENDERS = [
        r"^thanks?,?\s*$",
        r"^regards?,?\s*$",
        r"^best( regards)?,?\s*$",
        r"^kind regards?,?\s*$",
        r"^cheers,?\s*$",
        r"^sincerely,?\s*$",
        r"^best wishes,?\s*$",
        r"^warm regards?,?\s*$",
        r"^thank you,?\s*$",
        r"^many thanks,?\s*$",
    ]

    # Signature block patterns
    SIGNATURE_PATTERNS = [
        # Name followed by title/company
        r"^[A-Z][a-z]+ [A-Z][a-z]+\s*\n.*?(engineer|manager|director|analyst|consultant|developer)",
        # Phone/fax patterns
        r"(tel|phone|fax|cell|mobile|office):\s*[\+\d\-\(\)\s]+",
        # Common signature separators
        r"^[-_=]{3,}\s*$",  # --- or ___ or ===
        r"^\*{3,}\s*$",  # ***
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._disclaimer_pattern = re.compile(
            '|'.join(f'({p})' for p in self.DISCLAIMER_STARTERS),
            re.IGNORECASE | re.MULTILINE
        )
        self._ender_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.CONTENT_ENDERS
        ]

    def remove_disclaimers(self, text: str) -> Tuple[str, List[str]]:
        """
        Remove disclaimers from email text.

        Returns:
            Tuple of (cleaned_text, list_of_removed_sections)
        """
        if not text:
            return text, []

        removed = []
        lines = text.split('\n')
        result_lines = []

        skip_mode = False
        skip_start_idx = -1

        for i, line in enumerate(lines):
            # Check if this line starts a disclaimer
            if self._is_disclaimer_start(line):
                skip_mode = True
                skip_start_idx = i
                continue

            # If in skip mode, keep skipping until we hit something that looks like new content
            if skip_mode:
                # Check if this looks like new email content (From:, Subject:, etc.)
                if self._is_new_content_start(line):
                    skip_mode = False
                    if skip_start_idx >= 0:
                        removed.append(f"[Removed disclaimer at line {skip_start_idx}]")
                else:
                    continue

            result_lines.append(line)

        cleaned = '\n'.join(result_lines)

        # Also remove trailing signatures after "Thanks," etc.
        cleaned, sig_removed = self._trim_trailing_signature(cleaned)
        if sig_removed:
            removed.append("[Removed trailing signature]")

        return cleaned.strip(), removed

    def _is_disclaimer_start(self, line: str) -> bool:
        """Check if line starts a disclaimer block."""
        line = line.strip()
        if not line:
            return False
        return bool(self._disclaimer_pattern.search(line))

    def _is_new_content_start(self, line: str) -> bool:
        """Check if line indicates start of new email content."""
        line = line.strip()
        # Email header patterns
        if re.match(r'^(From|To|Cc|Subject|Date|Sent):', line, re.IGNORECASE):
            return True
        # Quote marker
        if line.startswith('>'):
            return True
        # Original message marker
        if re.match(r'^-+\s*original message\s*-+', line, re.IGNORECASE):
            return True
        return False

    def _trim_trailing_signature(self, text: str) -> Tuple[str, bool]:
        """Remove signature block at end of email."""
        lines = text.split('\n')

        # Find last "Thanks," / "Regards," type line
        cut_idx = -1
        for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
            line = lines[i].strip()
            for pattern in self._ender_patterns:
                if pattern.match(line):
                    cut_idx = i + 1  # Keep the "Thanks," line
                    break
            if cut_idx > 0:
                break

        if cut_idx > 0 and cut_idx < len(lines):
            # Check if remaining content is mostly signature (short lines, contact info)
            remaining = lines[cut_idx:]
            if self._looks_like_signature(remaining):
                return '\n'.join(lines[:cut_idx]), True

        return text, False

    def _looks_like_signature(self, lines: List[str]) -> bool:
        """Check if lines look like a signature block."""
        if not lines:
            return False

        # Count indicators
        contact_patterns = 0
        short_lines = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < 60:
                short_lines += 1
            if re.search(r'(tel|phone|fax|cell|email|@|www\.|http)', line, re.IGNORECASE):
                contact_patterns += 1

        # If mostly short lines with contact info, it's a signature
        total_lines = sum(1 for l in lines if l.strip())
        if total_lines == 0:
            return True
        if contact_patterns >= 2 or (short_lines / total_lines > 0.7):
            return True

        return False

    def remove_embedded_disclaimers(self, text: str) -> str:
        """
        Remove common disclaimer phrases that appear anywhere in text.

        This handles disclaimers embedded in forwarded/quoted email threads.
        """
        # Common disclaimer phrases to remove entirely
        embedded_patterns = [
            r"This electronic mail is intended only for the addressee\(s\)\.?\s*",
            r"This e?-?mail and any attachments? (are|is) confidential\.?\s*",
            r"If you (are not|have received this) .*?(delete|notify).*?[\.\n]",
            r"The information contained in this.*?privileged.*?[\.\n]",
            r"Any unauthorized.*?(prohibited|unlawful).*?[\.\n]",
            r"Please consider the environment before printing\.?\s*",
            r"Think before you print\.?\s*",
            r"Sent from my (iPhone|iPad|Android|BlackBerry|mobile|Samsung)\.?\s*",
            r"Get Outlook for (iOS|Android)\.?\s*",
        ]

        result = text
        for pattern in embedded_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)

        # Clean up multiple blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result


# Singleton instance
_remover = DisclaimerRemover()


def remove_disclaimers(text: str) -> str:
    """Remove disclaimers from text (convenience function)."""
    cleaned, _ = _remover.remove_disclaimers(text)
    # Also remove embedded disclaimers in forwarded content
    cleaned = _remover.remove_embedded_disclaimers(cleaned)
    return cleaned


def remove_disclaimers_with_info(text: str) -> Tuple[str, List[str]]:
    """Remove disclaimers and return info about what was removed."""
    cleaned, info = _remover.remove_disclaimers(text)
    # Also remove embedded disclaimers
    cleaned = _remover.remove_embedded_disclaimers(cleaned)
    return cleaned, info
