"""
Global Message Index for Cross-PST Thread Linking

Handles:
1. Duplicate detection across PST files
2. Missing message_id fallback
3. Thread relationship tracking
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MessageEntry:
    """Index entry for a single message."""
    message_id: str  # Original or generated
    record_ids: List[str] = field(default_factory=list)  # Can have multiple if duplicated
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    subject: str = ""
    sent_time: Optional[str] = None
    is_generated_id: bool = False  # True if message_id was generated (not from headers)


class MessageIndex:
    """
    Global index for message threading across multiple PST files.

    Ensures:
    - Unique message identification even without message_id header
    - Duplicate detection across PST files
    - Consistent thread linking
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "message_index.json"
        self.duplicates_path = self.index_dir / "duplicates.json"

        # In-memory index
        self.messages: Dict[str, MessageEntry] = {}  # message_id -> entry
        self.record_to_message: Dict[str, str] = {}  # record_id -> message_id
        self.duplicates: Dict[str, List[str]] = {}  # message_id -> [record_ids]

        self._load()

    def _load(self):
        """Load existing index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                for msg_id, entry_data in data.get('messages', {}).items():
                    self.messages[msg_id] = MessageEntry(**entry_data)
                self.record_to_message = data.get('record_to_message', {})

        if self.duplicates_path.exists():
            with open(self.duplicates_path, 'r') as f:
                self.duplicates = json.load(f)

        logger.info(f"Loaded message index: {len(self.messages)} messages")

    def save(self):
        """Save index to disk."""
        data = {
            'messages': {k: asdict(v) for k, v in self.messages.items()},
            'record_to_message': self.record_to_message,
            'updated_at': datetime.utcnow().isoformat()
        }
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

        with open(self.duplicates_path, 'w') as f:
            json.dump(self.duplicates, f, indent=2)

        logger.info(f"Saved message index: {len(self.messages)} messages, {len(self.duplicates)} duplicates")

    def generate_message_id(
        self,
        record_id: str,
        subject: str,
        sender: str,
        sent_time: str,
        body_hash: str
    ) -> str:
        """
        Generate a stable message_id for emails without one.

        Uses content-based hashing to ensure same email gets same ID.
        """
        # Create deterministic ID from content
        content = f"{subject}|{sender}|{sent_time}|{body_hash}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"<generated.{hash_value}@simplerag>"

    def register_message(
        self,
        record_id: str,
        message_id: Optional[str],
        in_reply_to: Optional[str],
        references: Optional[str],
        subject: str,
        sender: str,
        sent_time: Optional[str],
        body_text: str
    ) -> str:
        """
        Register a message in the index.

        Returns the (possibly generated) message_id.
        """
        # Generate message_id if missing
        is_generated = False
        if not message_id or message_id == 'None':
            body_hash = hashlib.md5(body_text.encode()).hexdigest()[:8] if body_text else "empty"
            message_id = self.generate_message_id(
                record_id, subject, sender, sent_time or "", body_hash
            )
            is_generated = True

        # Parse references into list
        ref_list = []
        if references:
            ref_list = [r.strip() for r in references.replace('>', '> ').split() if '@' in r]

        # Check for duplicate
        if message_id in self.messages:
            existing = self.messages[message_id]
            if record_id not in existing.record_ids:
                existing.record_ids.append(record_id)

                # Track as duplicate
                if message_id not in self.duplicates:
                    self.duplicates[message_id] = existing.record_ids.copy()
                else:
                    self.duplicates[message_id].append(record_id)

                logger.debug(f"Duplicate message detected: {message_id}")
        else:
            # New message
            self.messages[message_id] = MessageEntry(
                message_id=message_id,
                record_ids=[record_id],
                in_reply_to=in_reply_to,
                references=ref_list,
                subject=subject,
                sent_time=sent_time,
                is_generated_id=is_generated
            )

        # Map record to message
        self.record_to_message[record_id] = message_id

        return message_id

    def get_thread_chain(self, message_id: str) -> List[str]:
        """
        Get the full thread chain for a message.

        Returns list of message_ids from root to this message.
        """
        chain = []
        visited = set()
        current = message_id

        while current and current not in visited:
            visited.add(current)
            chain.append(current)

            entry = self.messages.get(current)
            if entry and entry.in_reply_to:
                current = entry.in_reply_to
            else:
                break

        return list(reversed(chain))

    def get_replies(self, message_id: str) -> List[str]:
        """Get all direct replies to a message."""
        replies = []
        for msg_id, entry in self.messages.items():
            if entry.in_reply_to == message_id:
                replies.append(msg_id)
        return replies

    def get_record_ids(self, message_id: str) -> List[str]:
        """Get all record_ids for a message (handles duplicates)."""
        entry = self.messages.get(message_id)
        return entry.record_ids if entry else []

    def get_message_id(self, record_id: str) -> Optional[str]:
        """Get message_id for a record."""
        return self.record_to_message.get(record_id)

    def get_stats(self) -> Dict:
        """Get index statistics."""
        generated_ids = sum(1 for e in self.messages.values() if e.is_generated_id)
        with_replies = sum(1 for e in self.messages.values() if e.in_reply_to)

        return {
            "total_messages": len(self.messages),
            "total_records": len(self.record_to_message),
            "duplicates": len(self.duplicates),
            "generated_ids": generated_ids,
            "with_in_reply_to": with_replies
        }
