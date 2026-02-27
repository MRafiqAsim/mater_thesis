"""
Attachment Content Processor

Extracts text content from email attachments for KG processing.
Supports: PDF, DOCX, XLSX, PPTX, TXT, CSV, RTF

Metadata is co-located alongside each binary file as {filename}.json
inside the classified subdirectory (knowledge_docs/, transactional_docs/, other_docs/).

Usage:
    processor = AttachmentProcessor(bronze_path="./data/bronze")
    content = processor.process_attachment(attachment_id, filename)

    # Or process all attachments for an email
    all_content = processor.get_email_attachment_content(email_id)
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import hashlib

from .document_parser import DocumentParser, ParsedDocument, DocumentType
from .attachment_classifier import AttachmentClassifier, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class AttachmentContent:
    """Extracted content from an attachment"""

    attachment_id: str
    filename: str
    email_id: str  # Parent email ID

    # Extracted content
    text: str
    doc_type: str
    page_count: int = 0

    # Metadata
    file_size: int = 0
    extraction_time: datetime = field(default_factory=datetime.now)
    extraction_success: bool = True
    error_message: str = ""

    # For KG processing
    has_tables: bool = False
    tables: List[Dict[str, Any]] = field(default_factory=list)

    # Classification (populated by AttachmentClassifier in Bronze layer)
    classification: str = ""  # "knowledge" | "transactional"
    classification_confidence: float = 0.0
    classification_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "email_id": self.email_id,
            "text": self.text,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "file_size": self.file_size,
            "extraction_time": self.extraction_time.isoformat(),
            "extraction_success": self.extraction_success,
            "error_message": self.error_message,
            "has_tables": self.has_tables,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "classification_signals": self.classification_signals,
            "char_count": len(self.text),
            "word_count": len(self.text.split()) if self.text else 0,
        }


class AttachmentProcessor:
    """
    Processes email attachments to extract text content.

    Works with attachments saved during PST extraction.
    Metadata JSON is co-located alongside each binary file in the
    classified subdirectory.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".pptx", ".ppt", ".txt", ".csv", ".rtf", ".html", ".htm"
    }

    def __init__(
        self,
        bronze_path: str,
        extract_tables: bool = True,
    ):
        """
        Initialize attachment processor.

        Args:
            bronze_path: Path to Bronze layer
            extract_tables: Extract tables from documents
        """
        self.bronze_path = Path(bronze_path)
        self.attachments_dir = self.bronze_path / "attachments"
        self.extract_tables = extract_tables

        # Initialize document parser
        self.parser = DocumentParser(extract_tables=extract_tables)

        # Initialize classifier for Bronze-layer classification
        self.classifier = AttachmentClassifier(bronze_path=bronze_path)

        # Classified subdirectories
        self.knowledge_dir = self.attachments_dir / "knowledge_docs"
        self.transactional_dir = self.attachments_dir / "transactional_docs"
        self.other_dir = self.attachments_dir / "other_docs"
        for d in [self.knowledge_dir, self.transactional_dir, self.other_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Legacy cache dir (read-only, for backward compat migration)
        self._legacy_cache_dir = self.bronze_path / "attachments_cache"

        # Statistics
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "cached": 0,
            "unsupported": 0,
        }

        # Load email-attachment mapping (also builds _email_file_paths index)
        self.email_attachments = self._load_attachment_mapping()

    def _load_attachment_mapping(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load mapping of emails to their attachments from email metadata.

        Also builds ``self._email_file_paths`` mapping email_id → Path to
        its JSON file, so we can enrich emails without re-scanning.

        Returns:
            Dict mapping email_id to list of attachment info
        """
        mapping = {}
        self._email_file_paths: Dict[str, Path] = {}
        emails_dir = self.bronze_path / "emails"

        if not emails_dir.exists():
            logger.warning(f"Emails directory not found: {emails_dir}")
            return mapping

        # Scan all email JSON files
        for email_file in emails_dir.rglob("*.json"):
            if "metadata" in email_file.parts:
                continue
            try:
                with open(email_file, "r", encoding="utf-8") as f:
                    email_data = json.load(f)

                email_id = email_data.get("message_id") or email_file.stem

                # Index the file path for later enrichment
                self._email_file_paths[email_id] = email_file

                has_attachments = email_data.get("has_attachments", False)
                attachment_count = email_data.get("attachment_count", 0)

                if has_attachments and attachment_count > 0:
                    # Get attachment info if available
                    attachments = email_data.get("attachments", [])
                    if attachments:
                        mapping[email_id] = attachments
                    else:
                        # Create placeholder if we know there are attachments
                        mapping[email_id] = [
                            {"filename": f"attachment_{i}", "attachment_id": f"{email_id}_{i}"}
                            for i in range(attachment_count)
                        ]

            except Exception as e:
                logger.debug(f"Error reading email file {email_file}: {e}")

        logger.info(f"Found {len(mapping)} emails with attachments")
        return mapping

    # ------------------------------------------------------------------
    # Co-located metadata (replaces the old attachments_cache/ directory)
    # ------------------------------------------------------------------

    def _get_metadata_path(
        self, attachment_id: str, filename: str, classification: str = ""
    ) -> Path:
        """Metadata JSON lives alongside the binary file in the classified dir."""
        if classification:
            dir_map = {
                "knowledge": self.knowledge_dir,
                "transactional": self.transactional_dir,
            }
            target_dir = dir_map.get(classification, self.other_dir)
        else:
            # Not yet classified — check classified dirs, then fall back to root
            for d in [self.knowledge_dir, self.transactional_dir, self.other_dir]:
                candidate = d / attachment_id / f"{filename}.json"
                if candidate.exists():
                    return candidate
            target_dir = self.attachments_dir
        return target_dir / attachment_id / f"{filename}.json"

    def _load_metadata(self, attachment_id: str, filename: str) -> Optional[AttachmentContent]:
        """Load extracted content from co-located metadata JSON.

        Search order:
        1. Classified dirs (knowledge_docs, transactional_docs, other_docs)
        2. Root attachments/{att_id}/{filename}.json
        3. Legacy attachments_cache/{att_id}.json (backward compat)
        """
        # 1 + 2: co-located metadata (classification="" triggers dir search)
        meta_path = self._get_metadata_path(attachment_id, filename)
        if meta_path.exists():
            return self._parse_metadata_json(meta_path)

        # 3: Legacy cache fallback
        legacy_path = self._legacy_cache_dir / f"{attachment_id}.json"
        if legacy_path.exists():
            return self._parse_metadata_json(legacy_path)

        return None

    def _parse_metadata_json(self, path: Path) -> Optional[AttachmentContent]:
        """Parse a metadata JSON file into an AttachmentContent."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return AttachmentContent(
                attachment_id=data["attachment_id"],
                filename=data["filename"],
                email_id=data["email_id"],
                text=data["text"],
                doc_type=data["doc_type"],
                page_count=data.get("page_count", 0),
                file_size=data.get("file_size", 0),
                extraction_success=data.get("extraction_success", True),
                error_message=data.get("error_message", ""),
                has_tables=data.get("has_tables", False),
                classification=data.get("classification", ""),
                classification_confidence=data.get("classification_confidence", 0.0),
                classification_signals=data.get("classification_signals", {}),
            )
        except Exception as e:
            logger.debug(f"Metadata load error for {path}: {e}")
            return None

    def _save_metadata(self, content: AttachmentContent) -> None:
        """Save extracted content as co-located JSON alongside the binary file."""
        meta_path = self._get_metadata_path(
            content.attachment_id, content.filename, content.classification
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(content.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug(f"Metadata save error for {content.attachment_id}: {e}")

    # ------------------------------------------------------------------
    # File storage helpers
    # ------------------------------------------------------------------

    def _store_classified_file(
        self,
        classification: str,
        attachment_id: str,
        filename: str,
        source_path: Path,
    ) -> None:
        """Move the original file into the classified subdirectory.

        After moving, removes the now-empty source directory (the
        root-level ``attachments/{att_id}/`` folder).
        """
        dir_map = {
            "knowledge": self.knowledge_dir,
            "transactional": self.transactional_dir,
        }
        target_dir = dir_map.get(classification, self.other_dir)
        dest = target_dir / attachment_id
        dest.mkdir(parents=True, exist_ok=True)
        dest_file = dest / filename
        if not dest_file.exists():
            shutil.move(str(source_path), str(dest_file))
            # Remove empty source directory left behind
            source_dir = source_path.parent
            if (source_dir != self.attachments_dir
                    and source_dir.exists()
                    and not any(source_dir.iterdir())):
                source_dir.rmdir()

    def find_attachment_file(
        self,
        attachment_id: str,
        filename: str
    ) -> Optional[Path]:
        """
        Find the attachment binary file on disk (skips .json metadata files).

        Checks (in order):
        1. Root location: attachments/{attachment_id}/{filename}
        2. Classified subdirs: knowledge_docs/, transactional_docs/, other_docs/
        3. Fallback glob in root (skip classified dirs)
        """
        # 1. Try root path (pre-classification location)
        direct_path = self.attachments_dir / attachment_id / filename
        if direct_path.exists() and not direct_path.suffix == ".json":
            return direct_path

        att_dir = self.attachments_dir / attachment_id
        if att_dir.exists() and att_dir.is_dir():
            files = [f for f in att_dir.iterdir() if not f.name.endswith(".json")]
            if files:
                return files[0]

        # 2. Try classified subdirectories (post-classification location)
        for classified_dir in [self.knowledge_dir, self.transactional_dir, self.other_dir]:
            classified_path = classified_dir / attachment_id / filename
            if classified_path.exists():
                return classified_path
            classified_att_dir = classified_dir / attachment_id
            if classified_att_dir.exists() and classified_att_dir.is_dir():
                files = [f for f in classified_att_dir.iterdir() if not f.name.endswith(".json")]
                if files:
                    return files[0]

        # 3. Fallback: glob in root (skip classified dirs and .json metadata)
        classified_dirs = {self.knowledge_dir, self.transactional_dir, self.other_dir}
        for path in self.attachments_dir.rglob(filename):
            if path.name.endswith(".json"):
                continue
            if any(path.is_relative_to(d) for d in classified_dirs):
                continue
            return path

        return None

    # ------------------------------------------------------------------
    # Email JSON enrichment
    # ------------------------------------------------------------------

    def _enrich_email_json(
        self,
        email_id: str,
        attachment_id: str,
        classification: str,
        confidence: float,
    ) -> None:
        """Write classification back into the email JSON's attachment entry."""
        email_file = self._email_file_paths.get(email_id)
        if not email_file or not email_file.exists():
            return

        try:
            with open(email_file, "r", encoding="utf-8") as f:
                email_data = json.load(f)

            for att in email_data.get("attachments", []):
                if att.get("attachment_id") == attachment_id:
                    att["classification"] = classification
                    att["classification_confidence"] = confidence
                    break
            else:
                return  # attachment not found in list

            with open(email_file, "w", encoding="utf-8") as f:
                json.dump(email_data, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            logger.debug(f"Email enrichment error for {email_id}/{attachment_id}: {e}")

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_attachment(
        self,
        attachment_id: str,
        filename: str,
        email_id: str,
        file_path: Optional[Path] = None
    ) -> AttachmentContent:
        """
        Process a single attachment and extract text.

        Args:
            attachment_id: Unique attachment identifier
            filename: Original filename
            email_id: Parent email ID
            file_path: Optional direct path to file

        Returns:
            AttachmentContent with extracted text
        """
        self.stats["processed"] += 1

        # Check co-located metadata first
        cached = self._load_metadata(attachment_id, filename)
        if cached:
            # Backward-compatible: re-classify old entries missing classification
            if not cached.classification:
                result = self.classifier.classify(cached)
                cached.classification = result.classification
                cached.classification_confidence = result.confidence
                cached.classification_signals = result.signals
                # Move file into classified dir
                original = self.find_attachment_file(attachment_id, filename)
                if original:
                    self._store_classified_file(
                        result.classification, attachment_id, filename, original
                    )
                self._save_metadata(cached)
                self._enrich_email_json(
                    email_id, attachment_id, result.classification, result.confidence
                )
            self.stats["cached"] += 1
            return cached

        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            self.stats["unsupported"] += 1
            return AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext,
                extraction_success=False,
                error_message=f"Unsupported file type: {ext}"
            )

        # Find the file
        if file_path is None:
            file_path = self.find_attachment_file(attachment_id, filename)

        if file_path is None or not file_path.exists():
            self.stats["failed"] += 1
            return AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext,
                extraction_success=False,
                error_message="Attachment file not found"
            )

        # Extract text using DocumentParser
        try:
            parsed = self.parser.parse(file_path)

            content = AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text=parsed.text,
                doc_type=parsed.doc_type.value,
                page_count=parsed.page_count,
                file_size=file_path.stat().st_size,
                extraction_success=True,
                has_tables=len(parsed.tables) > 0,
                tables=parsed.tables,
            )

            # Classify attachment using multi-signal scorer
            cls_result = self.classifier.classify(content)
            content.classification = cls_result.classification
            content.classification_confidence = cls_result.confidence
            content.classification_signals = cls_result.signals

            # Organize file into classified subdirectory
            self._store_classified_file(
                cls_result.classification, attachment_id, filename, file_path
            )

            self.stats["success"] += 1

            # Save co-located metadata
            self._save_metadata(content)

            # Enrich the parent email JSON with classification info
            self._enrich_email_json(
                email_id, attachment_id,
                cls_result.classification, cls_result.confidence
            )

            return content

        except Exception as e:
            self.stats["failed"] += 1
            logger.warning(f"Failed to extract {filename}: {e}")

            return AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext,
                extraction_success=False,
                error_message=str(e)
            )

    def get_email_attachment_content(
        self,
        email_id: str
    ) -> List[AttachmentContent]:
        """
        Get all attachment content for an email.

        Args:
            email_id: Email message ID

        Returns:
            List of AttachmentContent for all attachments
        """
        attachments = self.email_attachments.get(email_id, [])
        contents = []

        for att_info in attachments:
            att_id = att_info.get("attachment_id", "")
            filename = att_info.get("filename", "unknown")

            content = self.process_attachment(
                attachment_id=att_id,
                filename=filename,
                email_id=email_id
            )
            contents.append(content)

        return contents

    def get_combined_attachment_text(
        self,
        email_id: str,
        separator: str = "\n\n---ATTACHMENT---\n\n"
    ) -> str:
        """
        Get combined text from all attachments for an email.

        Args:
            email_id: Email message ID
            separator: Text separator between attachments

        Returns:
            Combined text from all attachments
        """
        contents = self.get_email_attachment_content(email_id)

        texts = []
        for content in contents:
            if content.extraction_success and content.text.strip():
                header = f"[Attachment: {content.filename}]\n"
                texts.append(header + content.text)

        return separator.join(texts)

    def process_all_attachments(
        self,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process all attachments in the Bronze layer.

        Args:
            progress_callback: Optional callback(count, message)

        Returns:
            Processing statistics
        """
        total_emails = len(self.email_attachments)

        for i, (email_id, attachments) in enumerate(self.email_attachments.items()):
            for att_info in attachments:
                att_id = att_info.get("attachment_id", "")
                filename = att_info.get("filename", "unknown")

                self.process_attachment(
                    attachment_id=att_id,
                    filename=filename,
                    email_id=email_id
                )

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, f"Processed {i + 1}/{total_emails} emails with attachments")

        return self.stats

    def scan_attachment_files(self) -> List[Dict[str, Any]]:
        """
        Scan the attachments directory to find all attachment files.

        Returns:
            List of attachment file info
        """
        files = []

        if not self.attachments_dir.exists():
            logger.warning(f"Attachments directory not found: {self.attachments_dir}")
            return files

        # Skip classified subdirectories (they contain copies)
        classified_dirs = {self.knowledge_dir, self.transactional_dir, self.other_dir}

        for file_path in self.attachments_dir.rglob("*"):
            if any(file_path.is_relative_to(d) for d in classified_dirs):
                continue
            if file_path.name.endswith(".json"):
                continue
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Derive attachment_id from directory structure
                att_id = file_path.parent.name if file_path.parent != self.attachments_dir else file_path.stem

                files.append({
                    "attachment_id": att_id,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "extension": file_path.suffix.lower(),
                })

        logger.info(f"Found {len(files)} attachment files")
        return files

    # ------------------------------------------------------------------
    # Legacy storage cleanup
    # ------------------------------------------------------------------

    def cleanup_legacy_storage(self) -> Dict[str, int]:
        """Remove legacy duplicate storage and migrate old cache to co-located metadata.

        Actions:
        1. Remove 32-char email_id directories (BronzeLayerLoader duplicates)
        2. Migrate old attachments_cache/*.json → co-located alongside classified files
        3. Remove empty root-level attachment directories
        4. Remove attachments_cache/ directory if empty

        Returns:
            Cleanup statistics
        """
        cleanup_stats = {
            "legacy_dirs_removed": 0,
            "cache_files_migrated": 0,
            "empty_dirs_removed": 0,
        }

        classified_dir_names = {"knowledge_docs", "transactional_docs", "other_docs"}

        # 1. Remove 32-char email_id directories (BronzeLayerLoader duplicates)
        if self.attachments_dir.exists():
            for child in sorted(self.attachments_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in classified_dir_names:
                    continue
                # PSTExtractor uses 12-char MD5 IDs; BronzeLayerLoader uses 32-char email IDs
                if len(child.name) >= 20:
                    shutil.rmtree(child)
                    cleanup_stats["legacy_dirs_removed"] += 1
                    logger.debug(f"Removed legacy dir: {child.name}")

        # 2. Migrate old attachments_cache/*.json → co-located metadata
        if self._legacy_cache_dir.exists():
            for cache_file in self._legacy_cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    att_id = data.get("attachment_id", "")
                    filename = data.get("filename", "")
                    classification = data.get("classification", "")

                    if att_id and filename and classification:
                        meta_path = self._get_metadata_path(att_id, filename, classification)
                        if not meta_path.exists():
                            meta_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(cache_file), str(meta_path))
                            cleanup_stats["cache_files_migrated"] += 1
                        else:
                            # Co-located version already exists, remove legacy
                            cache_file.unlink()
                            cleanup_stats["cache_files_migrated"] += 1
                    else:
                        # Unclassified cache entry — leave for now (will be re-processed)
                        logger.debug(f"Skipping unclassified cache: {cache_file.name}")

                except Exception as e:
                    logger.debug(f"Error migrating cache file {cache_file}: {e}")

        # 3. Remove empty root-level attachment directories
        if self.attachments_dir.exists():
            for child in sorted(self.attachments_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in classified_dir_names:
                    continue
                if not any(child.iterdir()):
                    child.rmdir()
                    cleanup_stats["empty_dirs_removed"] += 1

        # 4. Remove attachments_cache/ if empty
        if self._legacy_cache_dir.exists() and not any(self._legacy_cache_dir.iterdir()):
            self._legacy_cache_dir.rmdir()
            logger.info("Removed empty attachments_cache/ directory")

        logger.info(
            f"Cleanup complete: {cleanup_stats['legacy_dirs_removed']} legacy dirs removed, "
            f"{cleanup_stats['cache_files_migrated']} cache files migrated, "
            f"{cleanup_stats['empty_dirs_removed']} empty dirs removed"
        )
        return cleanup_stats


def extract_attachment_text(
    bronze_path: str,
    email_id: str
) -> str:
    """
    Convenience function to extract text from email attachments.

    Args:
        bronze_path: Path to Bronze layer
        email_id: Email message ID

    Returns:
        Combined text from all attachments
    """
    processor = AttachmentProcessor(bronze_path=bronze_path)
    return processor.get_combined_attachment_text(email_id)
