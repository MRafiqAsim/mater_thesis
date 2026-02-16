"""
Bronze Layer - Raw Ingestion

RULES (NON-NEGOTIABLE):
- Bronze data is IMMUTABLE and APPEND-ONLY
- Do NOT anonymize
- Do NOT summarize
- Do NOT normalize text
- Do NOT use AI or OCR
- Preserve original structure and ordering

Bronze must remain fully reproducible.
"""

import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field, asdict

# Email extraction
import email
from email import policy
from email.parser import BytesParser

# Document parsing (basic - no AI)
import mimetypes

logger = logging.getLogger(__name__)


@dataclass
class BronzeRecord:
    """A single Bronze layer record - raw, unprocessed data."""
    record_id: str
    source_file: str
    file_type: str
    ingestion_time: str
    file_size: int
    file_hash: str

    # Raw content
    raw_content: Optional[str] = None
    raw_binary_path: Optional[str] = None

    # Email-specific fields (if applicable)
    email_headers: Optional[Dict] = None
    email_body_text: Optional[str] = None
    email_body_html: Optional[str] = None
    email_attachments: List[Dict] = field(default_factory=list)
    email_thread_id: Optional[str] = None

    # Document-specific fields
    document_metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BronzeRecord':
        return cls(**data)


class BronzeIngestion:
    """
    Bronze Layer Ingestion - Raw data extraction only.

    Process:
    1. Detect new files in source_folder
    2. Atomically move to inprogress
    3. Extract raw content (NO AI, NO transformation)
    4. Write to bronze_data (append-only)
    5. Move to processed or error
    """

    def __init__(self, config):
        """
        Initialize Bronze ingestion.

        Args:
            config: SimpleRAGConfig with directory settings
        """
        self.config = config
        self.dirs = config.directories

        # Ensure directories exist
        self.dirs.ensure_directories()

        # Initialize PST extractor if available
        self._pst_extractor = None
        try:
            from pypff import file as pst_file
            self._has_pypff = True
        except ImportError:
            self._has_pypff = False
            logger.warning("pypff not available - PST extraction disabled")

        # Initialize global message index for cross-PST threading
        from .message_index import MessageIndex
        self.message_index = MessageIndex(self.dirs.bronze_dir / "metadata")

        logger.info(f"Bronze ingestion initialized: {self.dirs.source_folder}")

    def process_source_folder(self) -> List[BronzeRecord]:
        """
        Process all files in the source folder (including subdirectories).

        Returns:
            List of successfully ingested BronzeRecords
        """
        records = []

        # Find all files in source folder (recursive)
        source_files = list(self.dirs.source_folder.rglob("*"))
        supported = [f for f in source_files if f.is_file() and
                     f.suffix.lower() in self.config.supported_formats]

        logger.info(f"Found {len(supported)} supported files in source folder")

        for source_file in supported:
            try:
                file_records = self._process_file(source_file)
                records.extend(file_records)
            except Exception as e:
                logger.error(f"Failed to process {source_file}: {e}")
                self._move_to_error(source_file, str(e))

        # Save message index
        self.message_index.save()
        stats = self.message_index.get_stats()
        logger.info(f"Message index: {stats}")

        return records

    def _process_file(self, source_file: Path) -> List[BronzeRecord]:
        """Process a single file through Bronze ingestion."""
        logger.info(f"Processing: {source_file.name}")

        # Step 1: Atomically move to inprogress
        inprogress_file = self.dirs.inprogress_folder / source_file.name
        shutil.move(str(source_file), str(inprogress_file))

        try:
            # Step 2: Extract raw content based on file type
            file_type = source_file.suffix.lower()
            result = self._extract_raw(inprogress_file, file_type)

            # Handle both single record and list of records (PST returns list)
            records = result if isinstance(result, list) else [result] if result else []

            # Step 3: Write each record to Bronze (append-only)
            for record in records:
                if record:
                    self._write_to_bronze(record)

            # Step 4: Move to processed
            processed_file = self.dirs.processed_folder / source_file.name
            shutil.move(str(inprogress_file), str(processed_file))

            logger.info(f"Successfully ingested: {len(records)} records from {source_file.name}")
            return records

        except Exception as e:
            # Move to error folder
            self._move_to_error(inprogress_file, str(e))
            raise

    def _extract_raw(self, file_path: Path, file_type: str) -> Optional[BronzeRecord]:
        """
        Extract raw content from file.

        NO AI, NO transformation, NO normalization.
        """
        # Compute file hash for integrity
        with open(file_path, 'rb') as f:
            file_content = f.read()
            file_hash = hashlib.sha256(file_content).hexdigest()

        # Generate record ID
        record_id = f"bronze_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_hash[:8]}"

        # Base record
        record = BronzeRecord(
            record_id=record_id,
            source_file=str(file_path.name),
            file_type=file_type,
            ingestion_time=datetime.utcnow().isoformat(),
            file_size=len(file_content),
            file_hash=file_hash
        )

        # Extract based on type
        if file_type == ".pst":
            return self._extract_pst(file_path, record)
        elif file_type == ".msg":
            return self._extract_msg(file_path, record)
        elif file_type in [".pdf", ".docx", ".xlsx", ".pptx"]:
            return self._extract_document(file_path, record)
        elif file_type == ".txt":
            return self._extract_text(file_path, record)
        elif file_type in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self._extract_image(file_path, record)
        else:
            # Store raw binary
            return self._extract_binary(file_path, record)

    def _extract_pst(self, file_path: Path, record: BronzeRecord) -> List[BronzeRecord]:
        """
        Extract emails from PST file (raw, no transformation).

        Uses libpst (readpst) for reliable extraction with attachments.
        Falls back to pypff if libpst is not available.

        Creates ONE Bronze record PER EMAIL (not one for entire PST).
        Extracts and saves attachment binaries separately.
        """
        # Try libpst first (better attachment support)
        try:
            from .libpst_extractor import LibPSTExtractor

            extractor = LibPSTExtractor(
                output_dir=self.dirs.bronze_dir / "pst_extracted",
                bronze_dir=self.dirs.bronze_dir
            )

            # Extract PST
            extract_dir = extractor.extract_pst(file_path)

            # Parse into records
            records = extractor.parse_extracted_emails(extract_dir, file_path.name)

            logger.info(f"Extracted {len(records)} emails from PST using libpst")
            return records

        except FileNotFoundError as e:
            logger.warning(f"libpst not available: {e}, falling back to pypff")

        # Fallback to pypff
        if not self._has_pypff:
            raise ImportError("Neither libpst nor pypff available for PST extraction")

        from pypff import file as pst_file

        pst = pst_file()
        pst.open(str(file_path))

        # Extract all emails as separate records
        email_records = []
        self._extract_pst_folder_to_records(
            pst.root_folder,
            email_records,
            file_path.name,
            record.file_hash
        )

        pst.close()

        logger.info(f"Extracted {len(email_records)} emails from PST using pypff")
        return email_records

    def _extract_pst_folder_to_records(
        self,
        folder,
        records: List[BronzeRecord],
        pst_filename: str,
        pst_hash: str,
        path: str = ""
    ):
        """Recursively extract emails from PST folder into individual BronzeRecords."""
        current_path = f"{path}/{folder.name}" if path else folder.name

        # Extract messages from this folder
        for i in range(folder.number_of_sub_messages):
            try:
                message = folder.get_sub_message(i)
                record = self._extract_pst_message_to_record(
                    message, current_path, pst_filename, pst_hash, len(records)
                )
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Failed to extract message {i} from {current_path}: {e}")

        # Recurse into subfolders
        for i in range(folder.number_of_sub_folders):
            try:
                subfolder = folder.get_sub_folder(i)
                self._extract_pst_folder_to_records(
                    subfolder, records, pst_filename, pst_hash, current_path
                )
            except Exception as e:
                logger.warning(f"Failed to access subfolder {i}: {e}")

    def _extract_pst_message_to_record(
        self,
        message,
        folder_path: str,
        pst_filename: str,
        pst_hash: str,
        index: int
    ) -> Optional[BronzeRecord]:
        """Extract a single PST message into a BronzeRecord with attachments."""
        try:
            # Generate unique record ID
            message_id = getattr(message, 'internet_message_id', None) or f"msg_{index}"
            record_id = f"bronze_{pst_hash[:8]}_{hashlib.md5(str(message_id).encode()).hexdigest()[:8]}"

            # Extract thread-related headers
            transport_headers = getattr(message, 'transport_headers', None) or ""
            references = None
            thread_index = None

            # Parse References header from transport headers
            if transport_headers:
                for line in transport_headers.split('\n'):
                    if line.lower().startswith('references:'):
                        references = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('thread-index:'):
                        thread_index = line.split(':', 1)[1].strip()

            # Conversation ID for threading (Outlook specific)
            conversation_id = getattr(message, 'conversation_id', None)
            conversation_index = getattr(message, 'conversation_index', None)

            # Create record with full thread info
            record = BronzeRecord(
                record_id=record_id,
                source_file=pst_filename,
                file_type=".pst",
                ingestion_time=datetime.utcnow().isoformat(),
                file_size=0,  # Individual message size not easily available
                file_hash=pst_hash,
                email_headers={
                    "folder": folder_path,
                    "message_id": message_id,
                    "in_reply_to": getattr(message, 'in_reply_to_id', None),
                    "references": references,  # Chain of all ancestor message IDs
                    "thread_index": thread_index,  # Outlook thread index
                    "conversation_id": conversation_id,  # Outlook conversation ID
                    "conversation_index": str(conversation_index) if conversation_index else None,
                    "transport_headers": transport_headers
                },
                email_body_text=message.plain_text_body,
                email_body_html=message.html_body,
                email_thread_id=conversation_id,  # Store thread ID at top level too
                document_metadata={
                    "subject": message.subject,
                    "sender_name": message.sender_name,
                    "sender_email": getattr(message, 'sender_email_address', None),
                    "recipients": getattr(message, 'display_to', None),
                    "sent_time": str(message.delivery_time) if message.delivery_time else None,
                }
            )

            # Extract attachments
            record.email_attachments = self._extract_and_save_attachments(message, record_id)

            return record

        except Exception as e:
            logger.warning(f"Failed to extract message: {e}")
            return None

    def _extract_and_save_attachments(self, message, record_id: str) -> List[Dict]:
        """Extract and save attachment binaries from PST message."""
        attachments = []

        try:
            for i in range(message.number_of_attachments):
                try:
                    att = message.get_attachment(i)
                    att_name = att.name or f"attachment_{i}"
                    att_size = att.size or 0

                    # Get attachment data
                    att_data = None
                    try:
                        att_data = att.read_buffer(att_size) if att_size > 0 else None
                    except Exception:
                        pass

                    # Save attachment binary
                    att_path = None
                    if att_data:
                        att_dir = self.dirs.bronze_dir / "attachments" / record_id
                        att_dir.mkdir(parents=True, exist_ok=True)

                        # Sanitize filename
                        safe_name = "".join(c if c.isalnum() or c in '.-_' else '_' for c in att_name)
                        att_path = att_dir / safe_name

                        with open(att_path, 'wb') as f:
                            f.write(att_data)

                    attachments.append({
                        "name": att_name,
                        "size": att_size,
                        "content_type": getattr(att, 'mime_type', 'application/octet-stream'),
                        "saved_path": str(att_path) if att_path else None
                    })

                except Exception as e:
                    logger.warning(f"Failed to extract attachment {i}: {e}")

        except Exception as e:
            logger.warning(f"Failed to iterate attachments: {e}")

        return attachments

    def _extract_pst_folder(self, folder, emails: List, path: str = ""):
        """Legacy method - kept for compatibility."""
        current_path = f"{path}/{folder.name}" if path else folder.name

        for i in range(folder.number_of_sub_messages):
            try:
                message = folder.get_sub_message(i)
                email_data = self._extract_pst_message(message, current_path)
                if email_data:
                    emails.append(email_data)
            except Exception as e:
                logger.warning(f"Failed to extract message {i} from {current_path}: {e}")

        for i in range(folder.number_of_sub_folders):
            try:
                subfolder = folder.get_sub_folder(i)
                self._extract_pst_folder(subfolder, emails, current_path)
            except Exception as e:
                logger.warning(f"Failed to access subfolder {i}: {e}")

    def _extract_pst_message(self, message, folder_path: str) -> Optional[Dict]:
        """Extract raw email data from PST message."""
        try:
            # Raw extraction - no normalization
            return {
                "folder": folder_path,
                "subject": message.subject,
                "sender": message.sender_name,
                "sender_email": getattr(message, 'sender_email_address', None),
                "recipients": getattr(message, 'display_to', None),
                "sent_time": str(message.delivery_time) if message.delivery_time else None,
                "body_plain": message.plain_text_body,
                "body_html": message.html_body,
                "headers": getattr(message, 'transport_headers', None),
                "message_id": getattr(message, 'internet_message_id', None),
                "in_reply_to": getattr(message, 'in_reply_to_id', None),
                "attachments": self._extract_pst_attachments(message)
            }
        except Exception as e:
            logger.warning(f"Failed to extract message: {e}")
            return None

    def _extract_pst_attachments(self, message) -> List[Dict]:
        """Extract attachment metadata from PST message."""
        attachments = []
        try:
            for i in range(message.number_of_attachments):
                att = message.get_attachment(i)
                attachments.append({
                    "name": att.name,
                    "size": att.size,
                    "content_type": getattr(att, 'mime_type', 'application/octet-stream')
                })
        except Exception:
            pass
        return attachments

    def _extract_msg(self, file_path: Path, record: BronzeRecord) -> BronzeRecord:
        """Extract email from MSG file."""
        # Parse raw email
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        record.email_headers = dict(msg.items())
        record.email_body_text = msg.get_body(preferencelist=('plain',))
        if record.email_body_text:
            record.email_body_text = record.email_body_text.get_content()
        record.email_body_html = msg.get_body(preferencelist=('html',))
        if record.email_body_html:
            record.email_body_html = record.email_body_html.get_content()

        # Extract attachments metadata
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                record.email_attachments.append({
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "size": len(part.get_payload(decode=True) or b'')
                })

        return record

    def _extract_document(self, file_path: Path, record: BronzeRecord) -> BronzeRecord:
        """
        Extract document metadata (no content transformation).

        Actual text extraction happens in Silver layer via OCR/parsing.
        """
        record.document_metadata = {
            "mime_type": mimetypes.guess_type(str(file_path))[0],
            "requires_ocr": file_path.suffix.lower() == ".pdf",  # Mark for Silver
        }

        # Store binary reference
        binary_path = self.dirs.bronze_dir / "attachments" / f"{record.record_id}{file_path.suffix}"
        shutil.copy(str(file_path), str(binary_path))
        record.raw_binary_path = str(binary_path)

        return record

    def _extract_text(self, file_path: Path, record: BronzeRecord) -> BronzeRecord:
        """Extract text file content (raw, no normalization)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                record.raw_content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                record.raw_content = f.read()

        return record

    def _extract_image(self, file_path: Path, record: BronzeRecord) -> BronzeRecord:
        """
        Store image for later OCR in Silver layer.

        NO OCR at Bronze - just store the binary.
        """
        binary_path = self.dirs.bronze_dir / "attachments" / f"{record.record_id}{file_path.suffix}"
        shutil.copy(str(file_path), str(binary_path))
        record.raw_binary_path = str(binary_path)
        record.document_metadata = {
            "requires_ocr": True,
            "image_format": file_path.suffix.lower()
        }

        return record

    def _extract_binary(self, file_path: Path, record: BronzeRecord) -> BronzeRecord:
        """Store unknown binary file."""
        binary_path = self.dirs.bronze_dir / "attachments" / f"{record.record_id}{file_path.suffix}"
        shutil.copy(str(file_path), str(binary_path))
        record.raw_binary_path = str(binary_path)

        return record

    def _write_to_bronze(self, record: BronzeRecord):
        """
        Write record to Bronze layer (append-only).

        Bronze data is IMMUTABLE - never overwrite existing records.
        Also registers message in global index for cross-PST threading.
        """
        # Determine output path based on type
        if record.file_type in [".pst", ".msg"]:
            output_dir = self.dirs.bronze_dir / "emails"
        else:
            output_dir = self.dirs.bronze_dir / "documents"

        output_path = output_dir / f"{record.record_id}.json"

        # Check immutability - never overwrite
        if output_path.exists():
            logger.debug(f"Skipping existing record: {record.record_id}")
            return

        # Register in message index (handles missing message_id)
        if record.file_type in [".pst", ".msg"]:
            headers = record.email_headers or {}
            metadata = record.document_metadata or {}

            global_message_id = self.message_index.register_message(
                record_id=record.record_id,
                message_id=headers.get("message_id"),
                in_reply_to=headers.get("in_reply_to"),
                references=headers.get("references"),
                subject=metadata.get("subject", ""),
                sender=metadata.get("sender_name", ""),
                sent_time=metadata.get("sent_time"),
                body_text=record.email_body_text or ""
            )

            # Store the global message_id in headers
            if not headers.get("message_id") or headers.get("message_id") == "None":
                record.email_headers["message_id"] = global_message_id
                record.email_headers["message_id_generated"] = True

        # Write atomically
        temp_path = output_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        temp_path.rename(output_path)

        # Update ingestion log
        self._update_ingestion_log(record)

    def _update_ingestion_log(self, record: BronzeRecord):
        """Update the append-only ingestion log."""
        log_path = self.dirs.bronze_dir / "metadata" / "ingestion_log.jsonl"

        log_entry = {
            "record_id": record.record_id,
            "source_file": record.source_file,
            "file_type": record.file_type,
            "file_hash": record.file_hash,
            "ingestion_time": record.ingestion_time,
            "file_size": record.file_size
        }

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _move_to_error(self, file_path: Path, error_message: str):
        """Move failed file to error folder."""
        if not file_path.exists():
            return

        error_file = self.dirs.error_folder / file_path.name
        shutil.move(str(file_path), str(error_file))

        # Log error
        error_log = self.dirs.error_folder / "errors.jsonl"
        with open(error_log, 'a') as f:
            f.write(json.dumps({
                "file": file_path.name,
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }) + '\n')

    def list_bronze_records(self) -> Generator[BronzeRecord, None, None]:
        """Iterate over all Bronze records."""
        for subdir in ["emails", "documents"]:
            dir_path = self.dirs.bronze_dir / subdir
            if not dir_path.exists():
                continue

            for json_file in dir_path.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        yield BronzeRecord.from_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

    def get_bronze_record(self, record_id: str) -> Optional[BronzeRecord]:
        """Get a specific Bronze record by ID."""
        for subdir in ["emails", "documents"]:
            path = self.dirs.bronze_dir / subdir / f"{record_id}.json"
            if path.exists():
                with open(path, 'r') as f:
                    return BronzeRecord.from_dict(json.load(f))
        return None
