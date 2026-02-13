"""
PST Email Extractor

Extracts emails and attachments from Microsoft Outlook PST files.
Handles 35 years of email archives with proper encoding and error handling.
"""

import os
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Callable
from email.utils import parsedate_to_datetime
from .disclaimer_remover import remove_disclaimers

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """Email attachment"""

    filename: str
    content_type: str
    size: int
    content: bytes
    attachment_id: str = ""

    def __post_init__(self):
        if not self.attachment_id:
            self.attachment_id = hashlib.md5(
                f"{self.filename}:{self.size}".encode()
            ).hexdigest()[:12]

    def save(self, output_dir: str) -> str:
        """Save attachment to disk"""
        output_path = Path(output_dir) / self.attachment_id / self.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(self.content)

        return str(output_path)


@dataclass
class EmailMessage:
    """Extracted email message"""

    # Identification
    message_id: str
    source_pst: str
    folder_path: str

    # Headers
    subject: str
    sender: str
    sender_email: str
    recipients_to: List[str] = field(default_factory=list)
    recipients_cc: List[str] = field(default_factory=list)
    recipients_bcc: List[str] = field(default_factory=list)

    # Timestamps
    sent_time: Optional[datetime] = None
    received_time: Optional[datetime] = None

    # Content
    body_text: str = ""
    body_html: str = ""

    # Threading
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)

    # Attachments
    attachments: List[Attachment] = field(default_factory=list)
    has_attachments: bool = False

    # Metadata
    importance: str = "normal"
    is_read: bool = True
    categories: List[str] = field(default_factory=list)

    # Processing metadata
    extraction_time: datetime = field(default_factory=datetime.now)
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "source_pst": self.source_pst,
            "folder_path": self.folder_path,
            "subject": self.subject,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "recipients_to": self.recipients_to,
            "recipients_cc": self.recipients_cc,
            "sent_time": self.sent_time.isoformat() if self.sent_time else None,
            "received_time": self.received_time.isoformat() if self.received_time else None,
            "body_text": self.body_text,
            "conversation_id": self.conversation_id,
            "has_attachments": self.has_attachments,
            "attachment_count": len(self.attachments),
            "importance": self.importance,
            "language": self.language,
            "extraction_time": self.extraction_time.isoformat(),
        }

    def get_full_text(self) -> str:
        """Get full text content for processing"""
        parts = []

        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.sender:
            parts.append(f"From: {self.sender}")
        if self.recipients_to:
            parts.append(f"To: {', '.join(self.recipients_to)}")
        if self.sent_time:
            parts.append(f"Date: {self.sent_time.strftime('%Y-%m-%d %H:%M')}")

        parts.append("")  # Blank line
        parts.append(self.body_text)

        return "\n".join(parts)


class PSTExtractor:
    """
    Extract emails from PST files.

    Supports:
    - Microsoft Outlook PST files (97-2003, 2007+)
    - MBOX files (as fallback)
    - Recursive folder traversal
    - Attachment extraction
    - Encoding handling for international content
    """

    def __init__(
        self,
        extract_attachments: bool = True,
        attachment_output_dir: Optional[str] = None,
        max_attachment_size_mb: int = 50,
        supported_attachment_types: Optional[List[str]] = None
    ):
        """
        Initialize the PST extractor.

        Args:
            extract_attachments: Whether to extract attachments
            attachment_output_dir: Where to save attachments
            max_attachment_size_mb: Maximum attachment size to extract
            supported_attachment_types: List of extensions to extract (None = all)
        """
        self.extract_attachments = extract_attachments
        self.attachment_output_dir = attachment_output_dir
        self.max_attachment_size = max_attachment_size_mb * 1024 * 1024
        self.supported_attachment_types = supported_attachment_types or [
            ".pdf", ".docx", ".doc", ".xlsx", ".xls",
            ".pptx", ".ppt", ".txt", ".csv", ".rtf"
        ]

        # Statistics
        self.stats = {
            "total_emails": 0,
            "total_attachments": 0,
            "errors": 0,
            "folders_processed": 0,
        }

    def extract(
        self,
        pst_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Iterator[EmailMessage]:
        """
        Extract all emails from a PST file.

        Args:
            pst_path: Path to PST file
            progress_callback: Optional callback(count, status_message)

        Yields:
            EmailMessage objects
        """
        pst_path = Path(pst_path)

        if not pst_path.exists():
            raise FileNotFoundError(f"PST file not found: {pst_path}")

        logger.info(f"Starting extraction from: {pst_path}")

        # Try different extraction methods
        try:
            # Method 1: pypff (recommended for PST)
            yield from self._extract_with_pypff(pst_path, progress_callback)
        except ImportError:
            logger.warning("pypff not available, trying libpff-python")
            try:
                # Method 2: libpff
                yield from self._extract_with_libpff(pst_path, progress_callback)
            except ImportError:
                logger.warning("libpff not available, trying extract_msg")
                try:
                    # Method 3: extract_msg (for MSG files, convert PST first)
                    yield from self._extract_with_readpst(pst_path, progress_callback)
                except Exception as e:
                    raise RuntimeError(
                        f"No PST extraction library available. "
                        f"Install one of: pypff, libpff-python, or readpst. "
                        f"Error: {e}"
                    )

        logger.info(f"Extraction complete. Stats: {self.stats}")

    def _extract_with_pypff(
        self,
        pst_path: Path,
        progress_callback: Optional[Callable]
    ) -> Iterator[EmailMessage]:
        """Extract using pypff library"""
        import pypff

        pst = pypff.file()
        pst.open(str(pst_path))

        try:
            root = pst.get_root_folder()
            yield from self._process_folder_pypff(
                root,
                str(pst_path),
                "",
                progress_callback
            )
        finally:
            pst.close()

    def _process_folder_pypff(
        self,
        folder,
        pst_path: str,
        folder_path: str,
        progress_callback: Optional[Callable]
    ) -> Iterator[EmailMessage]:
        """Recursively process folders using pypff"""
        import pypff

        current_path = f"{folder_path}/{folder.name}" if folder_path else folder.name
        self.stats["folders_processed"] += 1

        # Process messages in this folder
        for i in range(folder.number_of_sub_messages):
            try:
                message = folder.get_sub_message(i)
                email = self._convert_pypff_message(message, pst_path, current_path)

                if email:
                    self.stats["total_emails"] += 1

                    if progress_callback and self.stats["total_emails"] % 100 == 0:
                        progress_callback(
                            self.stats["total_emails"],
                            f"Processing: {current_path}"
                        )

                    yield email

            except Exception as e:
                logger.warning(f"Error processing message {i} in {current_path}: {e}")
                self.stats["errors"] += 1

        # Process subfolders
        for i in range(folder.number_of_sub_folders):
            try:
                subfolder = folder.get_sub_folder(i)
                yield from self._process_folder_pypff(
                    subfolder, pst_path, current_path, progress_callback
                )
            except Exception as e:
                logger.warning(f"Error processing subfolder {i}: {e}")
                self.stats["errors"] += 1

    def _convert_pypff_message(
        self,
        message,
        pst_path: str,
        folder_path: str
    ) -> Optional[EmailMessage]:
        """Convert pypff message to EmailMessage"""
        try:
            # Generate message ID
            message_id = hashlib.md5(
                f"{pst_path}:{folder_path}:{message.subject or ''}:{message.delivery_time}".encode()
            ).hexdigest()

            # Parse sender
            sender = message.sender_name or ""
            sender_email = ""
            if hasattr(message, 'sender_email_address'):
                sender_email = message.sender_email_address or ""

            # Parse recipients
            recipients_to = self._parse_recipients(message, "to")
            recipients_cc = self._parse_recipients(message, "cc")

            # Parse body
            body_text = ""
            body_html = ""

            if hasattr(message, 'plain_text_body') and message.plain_text_body:
                body_text = self._decode_body(message.plain_text_body)
            if hasattr(message, 'html_body') and message.html_body:
                body_html = self._decode_body(message.html_body)

            # If no plain text, extract from HTML
            if not body_text and body_html:
                body_text = self._html_to_text(body_html)

            # Remove legal disclaimers (keeps thread info like From/Sent/To)
            if body_text:
                body_text = remove_disclaimers(body_text)

            # Parse timestamps
            sent_time = None
            received_time = None

            if hasattr(message, 'client_submit_time') and message.client_submit_time:
                sent_time = message.client_submit_time
            if hasattr(message, 'delivery_time') and message.delivery_time:
                received_time = message.delivery_time

            # Extract attachments
            attachments = []
            if self.extract_attachments and hasattr(message, 'number_of_attachments'):
                attachments = self._extract_attachments_pypff(message)

            return EmailMessage(
                message_id=message_id,
                source_pst=pst_path,
                folder_path=folder_path,
                subject=message.subject or "",
                sender=sender,
                sender_email=sender_email,
                recipients_to=recipients_to,
                recipients_cc=recipients_cc,
                sent_time=sent_time,
                received_time=received_time,
                body_text=body_text,
                body_html=body_html,
                conversation_id=getattr(message, 'conversation_topic', None),
                attachments=attachments,
                has_attachments=len(attachments) > 0,
                importance=self._parse_importance(message),
            )

        except Exception as e:
            logger.warning(f"Error converting message: {e}")
            return None

    def _parse_recipients(self, message, recipient_type: str) -> List[str]:
        """Parse recipients from message"""
        recipients = []

        try:
            if recipient_type == "to":
                # Try different attribute names
                for attr in ['display_to', 'to']:
                    if hasattr(message, attr):
                        value = getattr(message, attr)
                        if value:
                            recipients = [r.strip() for r in value.split(";") if r.strip()]
                            break
            elif recipient_type == "cc":
                for attr in ['display_cc', 'cc']:
                    if hasattr(message, attr):
                        value = getattr(message, attr)
                        if value:
                            recipients = [r.strip() for r in value.split(";") if r.strip()]
                            break
        except Exception:
            pass

        return recipients

    def _decode_body(self, body: Any) -> str:
        """Decode email body handling various encodings"""
        if body is None:
            return ""

        if isinstance(body, str):
            return body

        if isinstance(body, bytes):
            # Try common encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return body.decode(encoding)
                except (UnicodeDecodeError, AttributeError):
                    continue

            # Fallback: decode with replacement
            return body.decode('utf-8', errors='replace')

        return str(body)

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'head']):
                element.decompose()

            # Get text
            text = soup.get_text(separator='\n')

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            return '\n'.join(line for line in lines if line)

        except ImportError:
            # Fallback: basic regex cleaning
            import re
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _extract_attachments_pypff(self, message) -> List[Attachment]:
        """Extract attachments from pypff message"""
        attachments = []

        try:
            for i in range(message.number_of_attachments):
                try:
                    att = message.get_attachment(i)

                    filename = att.name or f"attachment_{i}"
                    size = att.size if hasattr(att, 'size') else 0

                    # Check size limit
                    if size > self.max_attachment_size:
                        logger.debug(f"Skipping large attachment: {filename} ({size} bytes)")
                        continue

                    # Check file type
                    ext = Path(filename).suffix.lower()
                    if self.supported_attachment_types and ext not in self.supported_attachment_types:
                        logger.debug(f"Skipping unsupported attachment type: {filename}")
                        continue

                    # Read content
                    content = att.read_buffer(size) if hasattr(att, 'read_buffer') else b""

                    attachment = Attachment(
                        filename=filename,
                        content_type=getattr(att, 'content_type', 'application/octet-stream'),
                        size=size,
                        content=content
                    )

                    # Save if output directory specified
                    if self.attachment_output_dir:
                        attachment.save(self.attachment_output_dir)

                    attachments.append(attachment)
                    self.stats["total_attachments"] += 1

                except Exception as e:
                    logger.warning(f"Error extracting attachment {i}: {e}")

        except Exception as e:
            logger.warning(f"Error accessing attachments: {e}")

        return attachments

    def _parse_importance(self, message) -> str:
        """Parse message importance/priority"""
        try:
            if hasattr(message, 'importance'):
                importance = message.importance
                if importance == 0:
                    return "low"
                elif importance == 2:
                    return "high"
        except Exception:
            pass
        return "normal"

    def _extract_with_libpff(
        self,
        pst_path: Path,
        progress_callback: Optional[Callable]
    ) -> Iterator[EmailMessage]:
        """Fallback extraction using libpff"""
        # Similar to pypff but using libpff bindings
        raise ImportError("libpff extraction not implemented")

    def _extract_with_readpst(
        self,
        pst_path: Path,
        progress_callback: Optional[Callable]
    ) -> Iterator[EmailMessage]:
        """Fallback extraction using readpst command-line tool"""
        import subprocess
        import tempfile
        import email
        from email import policy

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run readpst
            result = subprocess.run(
                ['readpst', '-e', '-o', temp_dir, str(pst_path)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"readpst failed: {result.stderr}")

            # Process extracted files
            for eml_path in Path(temp_dir).rglob("*.eml"):
                try:
                    with open(eml_path, 'rb') as f:
                        msg = email.message_from_binary_file(f, policy=policy.default)

                    yield self._convert_email_message(msg, str(pst_path), str(eml_path.parent))
                    self.stats["total_emails"] += 1

                except Exception as e:
                    logger.warning(f"Error processing {eml_path}: {e}")
                    self.stats["errors"] += 1

    def _convert_email_message(
        self,
        msg,
        pst_path: str,
        folder_path: str
    ) -> EmailMessage:
        """Convert standard email.message to EmailMessage"""
        # Generate ID
        message_id = msg.get('Message-ID', '') or hashlib.md5(
            f"{pst_path}:{msg.get('Subject', '')}:{msg.get('Date', '')}".encode()
        ).hexdigest()

        # Parse sender
        sender = msg.get('From', '')
        sender_email = ""
        if '<' in sender:
            sender_email = sender.split('<')[1].rstrip('>')
            sender = sender.split('<')[0].strip()

        # Parse date
        sent_time = None
        if msg.get('Date'):
            try:
                sent_time = parsedate_to_datetime(msg.get('Date'))
            except Exception:
                pass

        # Get body
        body_text = ""
        body_html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    body_text = part.get_content()
                elif content_type == 'text/html':
                    body_html = part.get_content()
        else:
            body_text = msg.get_content()

        if not body_text and body_html:
            body_text = self._html_to_text(body_html)

        # Remove legal disclaimers (keeps thread info like From/Sent/To)
        if body_text:
            body_text = remove_disclaimers(body_text)

        return EmailMessage(
            message_id=message_id,
            source_pst=pst_path,
            folder_path=folder_path,
            subject=msg.get('Subject', ''),
            sender=sender,
            sender_email=sender_email,
            recipients_to=[r.strip() for r in msg.get('To', '').split(',') if r.strip()],
            recipients_cc=[r.strip() for r in msg.get('Cc', '').split(',') if r.strip()],
            sent_time=sent_time,
            body_text=body_text,
            body_html=body_html,
            conversation_id=msg.get('Thread-Index'),
            in_reply_to=msg.get('In-Reply-To'),
        )

    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return self.stats.copy()


# Convenience function
def extract_pst(pst_path: str, output_dir: Optional[str] = None) -> List[EmailMessage]:
    """
    Extract all emails from a PST file.

    Args:
        pst_path: Path to PST file
        output_dir: Optional directory for attachments

    Returns:
        List of EmailMessage objects
    """
    extractor = PSTExtractor(
        extract_attachments=True,
        attachment_output_dir=output_dir
    )

    return list(extractor.extract(pst_path))
