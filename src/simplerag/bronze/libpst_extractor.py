"""
LibPST-based Email Extractor

Uses libpst/readpst output for reliable PST extraction with attachments.
"""

import os
import re
import email
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from email import policy
from email.parser import BytesParser

from .ingestion import BronzeRecord

logger = logging.getLogger(__name__)


class LibPSTExtractor:
    """
    Extract emails from PST using libpst (readpst command).

    More reliable than pypff for attachment extraction.
    """

    # Document types worth saving (not inline images/icons)
    USEFUL_EXTENSIONS = {
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.odt', '.ods', '.odp', '.rtf', '.txt', '.csv',
        # Archives
        '.zip', '.rar', '.7z', '.tar', '.gz',
        # Data files
        '.sql', '.json', '.xml', '.html', '.htm',
        # Other useful
        '.eml', '.msg',
    }

    # Minimum size for images to be considered attachments (not inline icons)
    MIN_IMAGE_SIZE = 50 * 1024  # 50KB - real photos/screenshots are usually larger

    # Skip these filenames (common inline image patterns)
    SKIP_PATTERNS = [
        'image001', 'image002', 'image003', 'image004', 'image005',
        'image006', 'image007', 'image008', 'image009', 'image010',
        'logo', 'icon', 'signature', 'banner', 'footer', 'header',
        'spacer', 'pixel', 'tracking', 'cid:'
    ]

    def __init__(self, output_dir: Path, bronze_dir: Path):
        self.output_dir = Path(output_dir)
        self.bronze_dir = Path(bronze_dir)
        self.readpst_path = self._find_readpst()

    def _find_readpst(self) -> str:
        """Find readpst executable."""
        # Try common locations
        paths = [
            "/opt/homebrew/bin/readpst",
            "/usr/local/bin/readpst",
            "/usr/bin/readpst"
        ]
        for p in paths:
            if os.path.exists(p):
                return p

        # Try which
        try:
            result = subprocess.run(["which", "readpst"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        raise FileNotFoundError("readpst not found. Install with: brew install libpst")

    def extract_pst(self, pst_path: Path) -> Path:
        """
        Extract PST file using readpst.

        Returns path to extracted directory.
        """
        pst_path = Path(pst_path)

        # Create output directory
        extract_dir = self.output_dir / pst_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting PST with libpst: {pst_path}")

        # Run readpst
        # -e = save attachments separately
        # -S = save emails as separate files
        # -o = output directory
        cmd = [
            self.readpst_path,
            "-e",  # Extract attachments
            "-S",  # Separate files
            "-o", str(extract_dir),
            str(pst_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"readpst failed: {result.stderr}")
            raise RuntimeError(f"readpst failed: {result.stderr}")

        logger.info(f"PST extracted to: {extract_dir}")
        return extract_dir

    def parse_extracted_emails(
        self,
        extract_dir: Path,
        pst_filename: str
    ) -> List[BronzeRecord]:
        """
        Parse extracted emails into BronzeRecords.

        Handles the libpst output format where:
        - Each email is a numbered file (1, 2, 3...)
        - Attachments are named like: 1-attachment.pdf
        - RTF bodies are: 1-rtf-body.rtf
        """
        extract_dir = Path(extract_dir)
        records = []

        # Compute PST hash for record IDs
        pst_hash = hashlib.md5(pst_filename.encode()).hexdigest()[:8]

        # Walk through all folders
        for folder_path in extract_dir.rglob("*"):
            if folder_path.is_dir():
                records.extend(
                    self._parse_folder(folder_path, pst_filename, pst_hash)
                )

        logger.info(f"Parsed {len(records)} emails from libpst extraction")
        return records

    def _parse_folder(
        self,
        folder_path: Path,
        pst_filename: str,
        pst_hash: str
    ) -> List[BronzeRecord]:
        """Parse emails from a single folder."""
        records = []

        # Find email files (numbered files without extension or with specific patterns)
        email_files = set()
        attachment_map: Dict[str, List[Path]] = {}  # email_num -> attachments

        for f in folder_path.iterdir():
            if f.is_file():
                name = f.name

                # Check if it's an email file (just a number)
                if re.match(r'^\d+$', name):
                    email_files.add(name)

                # Check if it's an attachment (number-something.ext)
                match = re.match(r'^(\d+)-(.+)$', name)
                if match:
                    email_num = match.group(1)
                    if email_num not in attachment_map:
                        attachment_map[email_num] = []
                    attachment_map[email_num].append(f)

        # Parse each email
        for email_num in sorted(email_files, key=int):
            email_path = folder_path / email_num
            attachments = attachment_map.get(email_num, [])

            try:
                record = self._parse_email_file(
                    email_path,
                    attachments,
                    folder_path.relative_to(folder_path.parents[1]) if len(folder_path.parents) > 1 else folder_path.name,
                    pst_filename,
                    pst_hash
                )
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse email {email_path}: {e}")

        return records

    def _parse_email_file(
        self,
        email_path: Path,
        attachment_paths: List[Path],
        folder_name: str,
        pst_filename: str,
        pst_hash: str
    ) -> Optional[BronzeRecord]:
        """Parse a single email file."""

        # Read email content
        with open(email_path, 'rb') as f:
            content = f.read()

        # Try to parse as email
        try:
            msg = BytesParser(policy=policy.default).parsebytes(content)
        except Exception:
            # Might be plain text
            msg = None

        # Extract headers
        if msg:
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            recipients = msg.get('To', '')
            date_str = msg.get('Date', '')
            message_id = msg.get('Message-ID', '')
            in_reply_to = msg.get('In-Reply-To', '')
            references = msg.get('References', '')

            # Get body
            body_text = ''
            body_html = ''

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        body_text = part.get_content()
                    elif content_type == 'text/html':
                        body_html = part.get_content()
            else:
                content_type = msg.get_content_type()
                if content_type == 'text/plain':
                    body_text = msg.get_content()
                elif content_type == 'text/html':
                    body_html = msg.get_content()
        else:
            # Plain text fallback
            try:
                body_text = content.decode('utf-8', errors='replace')
            except:
                body_text = str(content)

            subject = ''
            sender = ''
            recipients = ''
            date_str = ''
            message_id = ''
            in_reply_to = ''
            references = ''

        # Generate record ID
        content_hash = hashlib.md5(content).hexdigest()[:8]
        record_id = f"bronze_{pst_hash}_{content_hash}"

        # Process attachments
        saved_attachments = []
        for att_path in attachment_paths:
            # Skip RTF body files
            if att_path.name.endswith('-rtf-body.rtf'):
                continue

            saved_att = self._save_attachment(att_path, record_id)
            if saved_att:
                saved_attachments.append(saved_att)

        # Parse sent time
        sent_time = None
        if date_str:
            try:
                from email.utils import parsedate_to_datetime
                sent_time = str(parsedate_to_datetime(date_str))
            except:
                sent_time = date_str

        # Create record
        record = BronzeRecord(
            record_id=record_id,
            source_file=pst_filename,
            file_type=".pst",
            ingestion_time=datetime.utcnow().isoformat(),
            file_size=len(content),
            file_hash=content_hash,
            email_headers={
                "folder": str(folder_name),
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "references": references,
            },
            email_body_text=body_text if isinstance(body_text, str) else str(body_text),
            email_body_html=body_html if isinstance(body_html, str) else str(body_html),
            email_attachments=saved_attachments,
            document_metadata={
                "subject": subject,
                "sender_name": sender,
                "sender_email": self._extract_email(sender),
                "recipients": recipients,
                "sent_time": sent_time,
            }
        )

        return record

    def _extract_email(self, sender: str) -> Optional[str]:
        """Extract email address from sender string."""
        if not sender:
            return None
        match = re.search(r'<([^>]+)>', sender)
        if match:
            return match.group(1)
        if '@' in sender:
            return sender.strip()
        return None

    def _is_useful_attachment(self, filename: str, size: int) -> bool:
        """
        Check if attachment is useful (not inline image/icon).

        Only save:
        - Documents (PDF, Word, Excel, etc.)
        - Archives (ZIP, RAR)
        - Large images (likely screenshots/photos, not icons)
        """
        ext = Path(filename).suffix.lower()
        name_lower = filename.lower()

        # Always save useful document types
        if ext in self.USEFUL_EXTENSIONS:
            return True

        # Skip common inline image patterns
        for pattern in self.SKIP_PATTERNS:
            if pattern in name_lower:
                return False

        # For images, only save if large enough (real attachments, not icons)
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'}:
            if size >= self.MIN_IMAGE_SIZE:
                return True
            else:
                return False

        # Skip other small files
        if size < 1024:  # Less than 1KB
            return False

        return True

    def _save_attachment(self, att_path: Path, record_id: str) -> Optional[Dict]:
        """Save attachment to Bronze attachments directory (only useful ones)."""
        try:
            # Extract original filename (remove number- prefix)
            original_name = re.sub(r'^\d+-', '', att_path.name)
            size = att_path.stat().st_size

            # Check if this is a useful attachment
            if not self._is_useful_attachment(original_name, size):
                logger.debug(f"Skipping inline/icon attachment: {original_name} ({size} bytes)")
                return None

            # Create attachment directory
            att_dir = self.bronze_dir / "attachments" / record_id
            att_dir.mkdir(parents=True, exist_ok=True)

            # Copy attachment
            dest_path = att_dir / original_name
            shutil.copy2(att_path, dest_path)

            logger.debug(f"Saved useful attachment: {original_name} ({size} bytes)")

            return {
                "name": original_name,
                "size": size,
                "saved_path": str(dest_path),
                "content_type": self._guess_content_type(original_name)
            }
        except Exception as e:
            logger.warning(f"Failed to save attachment {att_path}: {e}")
            return None

    def _guess_content_type(self, filename: str) -> str:
        """Guess content type from filename."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
