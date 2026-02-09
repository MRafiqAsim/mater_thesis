"""
Email Loader (PST/MSG)
======================
Loaders for Microsoft Outlook email archives.

Supported formats:
- PST (Personal Storage Table) - Outlook archive files
- MSG (Outlook Message Format) - Individual email files

Features:
- Email metadata extraction (from, to, cc, date, subject)
- Email threading reconstruction
- Attachment extraction and linking
- Conversation grouping

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Optional, Dict, Any, Generator
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import os
import tempfile

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    """Structured email message representation."""
    message_id: str
    subject: str
    sender: str
    sender_email: str
    recipients: List[str]
    cc: List[str]
    bcc: List[str]
    date: Optional[datetime]
    body_text: str
    body_html: Optional[str]
    attachments: List[Dict[str, Any]]
    headers: Dict[str, str]
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        content = f"""Subject: {self.subject}
From: {self.sender} <{self.sender_email}>
To: {', '.join(self.recipients)}
Date: {self.date.isoformat() if self.date else 'Unknown'}

{self.body_text}
"""
        metadata = {
            "message_id": self.message_id,
            "subject": self.subject,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "recipients": self.recipients,
            "cc": self.cc,
            "date": self.date.isoformat() if self.date else None,
            "has_attachments": len(self.attachments) > 0,
            "attachment_count": len(self.attachments),
            "attachment_names": [a.get("filename", "") for a in self.attachments],
            "conversation_id": self.conversation_id,
            "in_reply_to": self.in_reply_to,
            "content_type": "email",
        }
        return Document(page_content=content, metadata=metadata)


@dataclass
class AttachmentInfo:
    """Information about an email attachment."""
    filename: str
    size_bytes: int
    content_type: str
    content: bytes
    parent_message_id: str


class MSGLoader:
    """
    Loader for Outlook MSG files.

    Usage:
        loader = MSGLoader()
        email = loader.load("/path/to/email.msg")
        doc = email.to_document()
    """

    def load(self, file_path: str) -> EmailMessage:
        """Load a single MSG file."""
        import extract_msg

        msg = extract_msg.Message(file_path)

        try:
            # Extract attachments
            attachments = []
            for att in msg.attachments:
                attachments.append({
                    "filename": att.longFilename or att.shortFilename or "unnamed",
                    "size_bytes": len(att.data) if att.data else 0,
                    "content_type": att.mimetype or "application/octet-stream",
                })

            # Parse date
            msg_date = None
            if msg.date:
                try:
                    msg_date = msg.date
                except Exception:
                    pass

            # Extract message ID for threading
            message_id = msg.messageId or hashlib.md5(
                f"{msg.subject}{msg.date}".encode()
            ).hexdigest()

            # Get threading headers
            in_reply_to = None
            references = []
            if hasattr(msg, 'header') and msg.header:
                in_reply_to = msg.header.get('In-Reply-To', '').strip('<>')
                refs = msg.header.get('References', '')
                references = [r.strip('<>') for r in refs.split() if r]

            return EmailMessage(
                message_id=message_id,
                subject=msg.subject or "(No Subject)",
                sender=msg.sender or "Unknown",
                sender_email=msg.senderEmail or "",
                recipients=[r.strip() for r in (msg.to or "").split(";") if r.strip()],
                cc=[r.strip() for r in (msg.cc or "").split(";") if r.strip()],
                bcc=[r.strip() for r in (msg.bcc or "").split(";") if r.strip()],
                date=msg_date,
                body_text=msg.body or "",
                body_html=msg.htmlBody,
                attachments=attachments,
                headers={},
                in_reply_to=in_reply_to,
                references=references,
            )

        finally:
            msg.close()

    def extract_attachments(self, file_path: str, output_dir: str) -> List[AttachmentInfo]:
        """Extract attachments from MSG file to directory."""
        import extract_msg

        msg = extract_msg.Message(file_path)
        attachments = []

        try:
            message_id = msg.messageId or hashlib.md5(
                f"{msg.subject}{msg.date}".encode()
            ).hexdigest()

            for att in msg.attachments:
                if att.data:
                    filename = att.longFilename or att.shortFilename or "unnamed"
                    output_path = Path(output_dir) / filename

                    # Handle duplicate filenames
                    counter = 1
                    while output_path.exists():
                        stem = output_path.stem
                        suffix = output_path.suffix
                        output_path = Path(output_dir) / f"{stem}_{counter}{suffix}"
                        counter += 1

                    with open(output_path, 'wb') as f:
                        f.write(att.data)

                    attachments.append(AttachmentInfo(
                        filename=str(output_path),
                        size_bytes=len(att.data),
                        content_type=att.mimetype or "application/octet-stream",
                        content=att.data,
                        parent_message_id=message_id,
                    ))

        finally:
            msg.close()

        return attachments


class PSTLoader:
    """
    Loader for Outlook PST archive files.

    Uses libpst for extraction. On Databricks, install via:
    %sh apt-get install -y pst-utils

    Usage:
        loader = PSTLoader()
        for email in loader.load_pst("/path/to/archive.pst"):
            doc = email.to_document()
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize PST loader.

        Args:
            temp_dir: Directory for temporary extraction (default: system temp)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def load_pst(self, pst_path: str) -> Generator[EmailMessage, None, None]:
        """
        Load all emails from a PST file.

        Args:
            pst_path: Path to PST file

        Yields:
            EmailMessage objects
        """
        import subprocess
        import shutil

        # Create extraction directory
        extract_dir = Path(self.temp_dir) / f"pst_extract_{hashlib.md5(pst_path.encode()).hexdigest()[:8]}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract PST using readpst (libpst)
            cmd = [
                "readpst",
                "-e",  # Extract each email to separate file
                "-o", str(extract_dir),  # Output directory
                "-q",  # Quiet mode
                pst_path
            ]

            logger.info(f"Extracting PST: {pst_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"readpst error: {result.stderr}")
                # Try alternative extraction with python library
                yield from self._load_pst_python(pst_path)
                return

            # Process extracted files
            msg_loader = MSGLoader()
            for root, dirs, files in os.walk(extract_dir):
                for filename in files:
                    file_path = Path(root) / filename

                    if filename.endswith('.eml') or filename.endswith('.msg'):
                        try:
                            email = self._parse_eml(str(file_path))
                            yield email
                        except Exception as e:
                            logger.warning(f"Failed to parse {file_path}: {e}")

        finally:
            # Cleanup
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

    def _load_pst_python(self, pst_path: str) -> Generator[EmailMessage, None, None]:
        """
        Fallback PST loading using pypff (Python library).
        Install: pip install pypff-python
        """
        try:
            import pypff
        except ImportError:
            logger.error("pypff not available. Install with: pip install pypff-python")
            return

        pst = pypff.file()
        pst.open(pst_path)

        try:
            root = pst.get_root_folder()
            yield from self._process_folder(root)
        finally:
            pst.close()

    def _process_folder(self, folder) -> Generator[EmailMessage, None, None]:
        """Recursively process PST folders."""
        # Process messages in current folder
        for i in range(folder.get_number_of_sub_messages()):
            try:
                message = folder.get_sub_message(i)
                yield self._message_to_email(message)
            except Exception as e:
                logger.warning(f"Failed to process message {i}: {e}")

        # Process subfolders
        for i in range(folder.get_number_of_sub_folders()):
            subfolder = folder.get_sub_folder(i)
            yield from self._process_folder(subfolder)

    def _message_to_email(self, message) -> EmailMessage:
        """Convert pypff message to EmailMessage."""
        subject = message.get_subject() or "(No Subject)"
        sender = message.get_sender_name() or "Unknown"

        # Get date
        msg_date = None
        try:
            msg_date = message.get_delivery_time()
        except Exception:
            pass

        return EmailMessage(
            message_id=hashlib.md5(f"{subject}{msg_date}".encode()).hexdigest(),
            subject=subject,
            sender=sender,
            sender_email=message.get_sender_email_address() or "",
            recipients=[],
            cc=[],
            bcc=[],
            date=msg_date,
            body_text=message.get_plain_text_body() or "",
            body_html=message.get_html_body(),
            attachments=[],
            headers={},
        )

    def _parse_eml(self, file_path: str) -> EmailMessage:
        """Parse EML (RFC 822) format email."""
        import email
        from email.utils import parsedate_to_datetime

        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)

        # Extract body
        body_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text = payload.decode('utf-8', errors='ignore')
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_text = payload.decode('utf-8', errors='ignore')

        # Parse date
        msg_date = None
        date_str = msg.get('Date')
        if date_str:
            try:
                msg_date = parsedate_to_datetime(date_str)
            except Exception:
                pass

        # Get message ID
        message_id = msg.get('Message-ID', '').strip('<>')
        if not message_id:
            message_id = hashlib.md5(f"{msg.get('Subject')}{date_str}".encode()).hexdigest()

        return EmailMessage(
            message_id=message_id,
            subject=msg.get('Subject', '(No Subject)'),
            sender=msg.get('From', 'Unknown'),
            sender_email=msg.get('From', ''),
            recipients=[r.strip() for r in msg.get('To', '').split(',') if r.strip()],
            cc=[r.strip() for r in msg.get('Cc', '').split(',') if r.strip()],
            bcc=[r.strip() for r in msg.get('Bcc', '').split(',') if r.strip()],
            date=msg_date,
            body_text=body_text,
            body_html=None,
            attachments=[],
            headers=dict(msg.items()),
            in_reply_to=msg.get('In-Reply-To', '').strip('<>'),
            references=[r.strip('<>') for r in msg.get('References', '').split() if r],
        )


class EmailThreader:
    """
    Reconstruct email conversation threads.

    Uses Message-ID, In-Reply-To, and References headers to build
    conversation trees.
    """

    def __init__(self):
        self.messages: Dict[str, EmailMessage] = {}
        self.threads: Dict[str, List[str]] = {}  # conversation_id -> [message_ids]

    def add_message(self, email: EmailMessage) -> str:
        """
        Add message and determine its thread.

        Returns:
            conversation_id for the thread
        """
        self.messages[email.message_id] = email

        # Find existing thread
        conversation_id = None

        # Check In-Reply-To
        if email.in_reply_to and email.in_reply_to in self.messages:
            parent = self.messages[email.in_reply_to]
            conversation_id = parent.conversation_id

        # Check References
        if not conversation_id:
            for ref in email.references:
                if ref in self.messages:
                    parent = self.messages[ref]
                    conversation_id = parent.conversation_id
                    break

        # Create new thread if needed
        if not conversation_id:
            conversation_id = email.message_id

        # Update email and thread
        email.conversation_id = conversation_id
        if conversation_id not in self.threads:
            self.threads[conversation_id] = []
        self.threads[conversation_id].append(email.message_id)

        return conversation_id

    def get_thread(self, conversation_id: str) -> List[EmailMessage]:
        """Get all messages in a thread, sorted by date."""
        if conversation_id not in self.threads:
            return []

        messages = [self.messages[mid] for mid in self.threads[conversation_id]]
        return sorted(messages, key=lambda m: m.date or datetime.min)

    def get_all_threads(self) -> Dict[str, List[EmailMessage]]:
        """Get all conversation threads."""
        return {
            cid: self.get_thread(cid)
            for cid in self.threads
        }


# Export
__all__ = ['MSGLoader', 'PSTLoader', 'EmailMessage', 'EmailThreader', 'AttachmentInfo']
