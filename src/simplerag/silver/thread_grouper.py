"""
Thread Grouper for Email Conversations

Groups emails into threads using:
1. conversation_id (Outlook specific)
2. in_reply_to / references headers (standard)
3. Subject matching (fallback)
"""

import re
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

from ..bronze.ingestion import BronzeRecord

logger = logging.getLogger(__name__)


@dataclass
class EmailThread:
    """A group of related emails forming a conversation thread."""
    thread_id: str
    subject: str
    emails: List[BronzeRecord] = field(default_factory=list)
    participants: Set[str] = field(default_factory=set)

    # Thread metadata
    first_email_time: Optional[str] = None
    last_email_time: Optional[str] = None
    email_count: int = 0

    def add_email(self, record: BronzeRecord):
        """Add an email to this thread."""
        self.emails.append(record)
        self.email_count = len(self.emails)

        # Update participants
        metadata = record.document_metadata or {}
        if metadata.get("sender_email"):
            self.participants.add(metadata["sender_email"])
        if metadata.get("sender_name"):
            self.participants.add(metadata["sender_name"])

        # Update timestamps
        sent_time = metadata.get("sent_time")
        if sent_time:
            if not self.first_email_time or sent_time < self.first_email_time:
                self.first_email_time = sent_time
            if not self.last_email_time or sent_time > self.last_email_time:
                self.last_email_time = sent_time

    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "email_count": self.email_count,
            "participants": list(self.participants),
            "first_email_time": self.first_email_time,
            "last_email_time": self.last_email_time,
            "email_ids": [e.record_id for e in self.emails]
        }


class ThreadGrouper:
    """
    Groups emails into conversation threads.

    Uses multiple strategies:
    1. Outlook conversation_id (most reliable)
    2. In-Reply-To and References headers (standard)
    3. Subject matching with "Re:", "Fwd:" normalization (fallback)
    """

    # Patterns to remove from subjects for matching
    SUBJECT_PREFIXES = re.compile(r'^(re|fw|fwd|aw|wg|r|回复|转发):\s*', re.IGNORECASE)

    def __init__(self):
        # Mapping structures
        self.message_id_to_thread: Dict[str, str] = {}  # message_id -> thread_id
        self.threads: Dict[str, EmailThread] = {}  # thread_id -> EmailThread
        self.subject_to_thread: Dict[str, str] = {}  # normalized_subject -> thread_id

    def group_emails(self, records: List[BronzeRecord]) -> List[EmailThread]:
        """
        Group Bronze email records into threads.

        Returns list of EmailThread objects.
        """
        # Sort by sent time
        def get_sent_time(r):
            meta = r.document_metadata or {}
            return meta.get("sent_time") or ""

        sorted_records = sorted(records, key=get_sent_time)

        for record in sorted_records:
            self._assign_to_thread(record)

        # Sort threads by first email time
        sorted_threads = sorted(
            self.threads.values(),
            key=lambda t: t.first_email_time or ""
        )

        logger.info(f"Grouped {len(sorted_records)} emails into {len(sorted_threads)} threads")
        return sorted_threads

    def _assign_to_thread(self, record: BronzeRecord):
        """Assign an email to a thread."""
        headers = record.email_headers or {}
        metadata = record.document_metadata or {}

        message_id = headers.get("message_id")
        in_reply_to = headers.get("in_reply_to")
        references = headers.get("references")
        conversation_id = headers.get("conversation_id")
        subject = metadata.get("subject", "")

        thread_id = None

        # Strategy 1: Use Outlook conversation_id
        if conversation_id:
            thread_id = f"conv_{conversation_id}"

        # Strategy 2: Check if this is a reply (in_reply_to)
        if not thread_id and in_reply_to:
            parent_thread = self.message_id_to_thread.get(in_reply_to)
            if parent_thread:
                thread_id = parent_thread

        # Strategy 3: Check References header
        if not thread_id and references:
            for ref_id in references.split():
                ref_id = ref_id.strip('<>')
                if ref_id in self.message_id_to_thread:
                    thread_id = self.message_id_to_thread[ref_id]
                    break

        # Strategy 4: Subject matching
        if not thread_id:
            normalized_subject = self._normalize_subject(subject)
            if normalized_subject in self.subject_to_thread:
                thread_id = self.subject_to_thread[normalized_subject]

        # Create new thread if not found
        if not thread_id:
            thread_id = f"thread_{len(self.threads)}"

        # Create or get thread
        if thread_id not in self.threads:
            self.threads[thread_id] = EmailThread(
                thread_id=thread_id,
                subject=subject
            )

        # Add email to thread
        thread = self.threads[thread_id]
        thread.add_email(record)

        # Update mappings
        if message_id:
            self.message_id_to_thread[message_id] = thread_id

        normalized_subject = self._normalize_subject(subject)
        if normalized_subject and normalized_subject not in self.subject_to_thread:
            self.subject_to_thread[normalized_subject] = thread_id

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject for matching (remove Re:, Fwd:, etc.)."""
        if not subject:
            return ""

        # Remove prefix patterns
        normalized = self.SUBJECT_PREFIXES.sub('', subject.strip())

        # Recursively remove prefixes
        while normalized != subject:
            subject = normalized
            normalized = self.SUBJECT_PREFIXES.sub('', subject.strip())

        return normalized.lower().strip()

    def get_thread(self, thread_id: str) -> Optional[EmailThread]:
        """Get a specific thread by ID."""
        return self.threads.get(thread_id)

    def get_thread_for_message(self, message_id: str) -> Optional[EmailThread]:
        """Get the thread containing a specific message."""
        thread_id = self.message_id_to_thread.get(message_id)
        if thread_id:
            return self.threads.get(thread_id)
        return None
