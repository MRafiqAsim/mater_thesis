"""
Thread-Aware Silver Processor

Processes email threads with semantic context preservation:
1. Groups emails into conversation threads
2. Concatenates thread emails chronologically
3. Chunks with thread boundaries respected
4. Generates thread summaries for high-level retrieval
5. Maintains consistent anonymization across thread
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.thread_grouper import ThreadGrouper, EmailThread
from ingestion.chunker import SemanticChunker, Chunk
from ingestion.language_detector import LanguageDetector
from ingestion.attachment_processor import AttachmentProcessor
from ingestion.email_text_cleaner import clean_email_text
from anonymization.pii_detector import PIIDetector
from anonymization.anonymizer import Anonymizer, AnonymizationStrategy
from extraction.kg_entity_extractor import (
    KGEntityExtractor,
    KGEntity,
    SpaCyKGExtractor,
    create_kg_extractor,
)
from extraction.relationship_extractor import (
    RelationshipExtractor,
    create_relationship_extractor,
)

logger = logging.getLogger(__name__)


@dataclass
class ThreadChunk:
    """A chunk from a thread with context"""

    chunk_id: str
    thread_id: str
    chunk_index: int

    # Content
    text_original: str
    text_anonymized: str
    token_count: int

    # Thread context
    thread_subject: str
    thread_participants: List[str]
    thread_email_count: int
    email_position: str  # e.g., "1/5", "3/5"

    # PII (for anonymization)
    pii_entities: List[Dict[str, Any]] = field(default_factory=list)
    pii_count: int = 0

    # Knowledge Graph Entities (for PathRAG)
    # These are non-PII entities useful for building knowledge graphs
    kg_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Knowledge Graph Relationships (for PathRAG)
    # Relationships between entities: (source, target, description, keywords, weight)
    kg_relationships: List[Dict[str, Any]] = field(default_factory=list)

    # Attachment information
    has_attachments: bool = False
    attachment_count: int = 0
    attachment_filenames: List[str] = field(default_factory=list)

    # Source tracking
    source_type: str = "email"                    # "email" | "attachment"
    source_attachment_filename: str = ""           # set when source_type == "attachment"
    attachment_classification: str = ""            # "knowledge" | "transactional" | ""
    classification_confidence: float = 0.0        # 0.0–1.0 from Bronze classifier

    # Metadata
    language: str = "en"
    processing_mode: str = "local"
    processing_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "thread_id": self.thread_id,
            "chunk_index": self.chunk_index,
            "text_original": self.text_original,
            "text_anonymized": self.text_anonymized,
            "token_count": self.token_count,
            "thread_subject": self.thread_subject,
            "thread_participants": self.thread_participants,
            "thread_email_count": self.thread_email_count,
            "email_position": self.email_position,
            "pii_entities": self.pii_entities,
            "pii_count": self.pii_count,
            "kg_entities": self.kg_entities,
            "kg_relationships": self.kg_relationships,
            "has_attachments": self.has_attachments,
            "attachment_count": self.attachment_count,
            "attachment_filenames": self.attachment_filenames,
            "source_type": self.source_type,
            "source_attachment_filename": self.source_attachment_filename,
            "attachment_classification": self.attachment_classification,
            "classification_confidence": self.classification_confidence,
            "language": self.language,
            "processing_mode": self.processing_mode,
            "processing_time": self.processing_time.isoformat(),
        }


@dataclass
class ThreadSummary:
    """Summary of an email thread for high-level retrieval"""

    thread_id: str
    subject: str
    participants: List[str]
    email_count: int
    date_range: str
    summary: str
    key_topics: List[str]
    chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "participants": self.participants,
            "email_count": self.email_count,
            "date_range": self.date_range,
            "summary": self.summary,
            "key_topics": self.key_topics,
            "chunk_ids": self.chunk_ids,
        }


class ThreadAwareProcessor:
    """
    Process email threads with semantic context preservation.

    Pipeline:
    1. Load emails from Bronze layer
    2. Group into conversation threads
    3. Concatenate thread emails (chronological)
    4. Detect language
    5. Chunk with thread boundaries
    6. Detect and anonymize PII (consistent across thread)
    7. Generate thread summaries
    8. Save to Silver layer
    """

    def __init__(
        self,
        bronze_path: str,
        silver_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        anonymization_strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        confidence_threshold: float = 0.5,
        generate_summaries: bool = True,
        openai_api_key: Optional[str] = None,
        kg_extractor_strategy: str = "spacy",
        kg_extractor: Optional[KGEntityExtractor] = None,
        relationship_extractor_strategy: str = "cooccurrence",
        relationship_extractor: Optional[RelationshipExtractor] = None,
        extract_relationships: bool = True,
        # Azure OpenAI settings
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: Optional[str] = None,
        # Attachment processing
        process_attachments: bool = True,
        include_attachment_text: bool = True,
        # Processing mode and identity
        processing_mode: str = "local",
        identity_registry=None,
    ):
        """
        Initialize thread-aware processor.

        Args:
            bronze_path: Path to Bronze layer
            silver_path: Path to Silver layer output
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            anonymization_strategy: Strategy for PII anonymization
            confidence_threshold: Minimum confidence for PII detection
            generate_summaries: Whether to generate thread summaries
            openai_api_key: OpenAI/Azure API key for LLM features
            kg_extractor_strategy: Strategy for KG extraction ("spacy", "llm", "hybrid")
            kg_extractor: Optional custom KG extractor (overrides strategy)
            relationship_extractor_strategy: Strategy for relationship extraction
                                             ("cooccurrence", "llm", "hybrid")
            relationship_extractor: Optional custom relationship extractor
            extract_relationships: Whether to extract relationships (default: True)
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name
            process_attachments: Whether to process email attachments
            include_attachment_text: Include attachment text in chunks for KG extraction
            processing_mode: Pipeline mode ("local", "llm", "hybrid")
            identity_registry: Optional IdentityRegistry for consistent pseudonyms
        """
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.generate_summaries = generate_summaries
        self.openai_api_key = openai_api_key
        self.processing_mode = processing_mode
        self.identity_registry = identity_registry

        # Initialize components
        self.thread_grouper = ThreadGrouper()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.language_detector = LanguageDetector()
        self.pii_detector = PIIDetector(
            confidence_threshold=confidence_threshold,
            identity_registry=identity_registry,
        )
        self.anonymizer = Anonymizer(
            detector=self.pii_detector,
            strategy=anonymization_strategy,
            consistent_replacement=True,  # Important: same entity = same placeholder
            identity_registry=identity_registry,
        )

        # Attachment processing
        self.process_attachments = process_attachments
        self.include_attachment_text = include_attachment_text

        # Attachment classification limits (configurable via env)
        self.max_tokens_knowledge = int(os.environ.get("ATTACHMENT_MAX_TOKENS_KNOWLEDGE", "0"))
        self.attachment_processor = None

        if process_attachments:
            try:
                self.attachment_processor = AttachmentProcessor(
                    bronze_path=str(self.bronze_path),
                    extract_tables=True,
                )
                logger.info(f"AttachmentProcessor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AttachmentProcessor: {e}")
                self.process_attachments = False

        # Store Azure settings
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment

        # Initialize KG extractor (modular for benchmarking)
        if kg_extractor:
            self.kg_extractor = kg_extractor
        else:
            self.kg_extractor = create_kg_extractor(
                strategy=kg_extractor_strategy,
                languages=["en", "nl"],
                openai_api_key=openai_api_key,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment
            )
        logger.info(f"Using KG extractor: {self.kg_extractor.name}")

        # Initialize relationship extractor (modular for benchmarking)
        self.extract_relationships = extract_relationships
        if extract_relationships:
            if relationship_extractor:
                self.relationship_extractor = relationship_extractor
            else:
                self.relationship_extractor = create_relationship_extractor(
                    strategy=relationship_extractor_strategy,
                    openai_api_key=openai_api_key,
                    use_azure=use_azure,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    azure_deployment=azure_deployment
                )
            logger.info(f"Using relationship extractor: {self.relationship_extractor.name}")
        else:
            self.relationship_extractor = None

        # Create directories
        self._create_directories()

        # Statistics
        self.stats = {
            "threads_processed": 0,
            "single_emails": 0,
            "multi_email_threads": 0,
            "chunks_created": 0,
            "summaries_generated": 0,
            "pii_detected": 0,
            "kg_entities_extracted": 0,
            "kg_relationships_extracted": 0,
            "attachments_processed": 0,
            "attachments_with_text": 0,
            "attachments_skipped_non_knowledge": 0,
            "attachment_chunks_created": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def _extract_kg_entities(self, text: str, language: str) -> Tuple[List[Dict[str, Any]], List[KGEntity]]:
        """
        Extract knowledge graph entities using modular extractor.

        Delegates to configured KG extractor (spaCy, LLM, or hybrid).
        Returns tuple of (entity_dicts, raw_entities) for PathRAG.
        """
        entities = self.kg_extractor.extract(text, language)
        entity_dicts = [e.to_dict() for e in entities]
        self.stats["kg_entities_extracted"] += len(entity_dicts)
        return entity_dicts, entities

    def _get_person_pseudonym(self, name: str) -> str:
        """
        Get a pseudonym for a person name.

        Lookup order:
        1. Identity registry — known email senders/recipients (exact + fuzzy)
        2. Anonymizer's value mapping — consistent within processing run
        3. Anonymizer pass — generates new PERSON_N and caches it

        Known people (registry) get stable pseudonyms tied to their email.
        NER-only names get consistent pseudonyms via the anonymizer's
        _value_mapping (same text → same pseudonym within a run).
        """
        # 1. Identity registry (email-based, stable across runs)
        if self.identity_registry:
            pseudonym = self.identity_registry.get_pseudonym(name)
            if pseudonym:
                return pseudonym

        # 2. Anonymizer's existing mapping (consistent within run)
        mapped = self.anonymizer._value_mapping.get(name, "")
        if mapped:
            return mapped.strip("[]")

        # 3. Run anonymizer — detects as PERSON PII, generates & caches pseudonym
        anon_result = self.anonymizer.anonymize(name, "en")
        if anon_result.anonymized_text != name:
            return anon_result.anonymized_text.strip("[]")

        # 4. Anonymizer didn't detect it as PII — force a consistent pseudonym
        if not hasattr(self, "_ner_person_counter"):
            self._ner_person_counter = 900  # high offset to avoid collision
        self._ner_person_counter += 1
        pseudonym = f"PERSON_{self._ner_person_counter:03d}"
        self.anonymizer._value_mapping[name] = f"[{pseudonym}]"
        return pseudonym

    def _anonymize_kg_entities(self, entity_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace PERSON entity text with pseudonyms in KG entity dicts.

        All PERSON entities are anonymized regardless of is_pii flag,
        since person names in the knowledge graph are always PII.
        """
        anonymized = []
        for e in entity_dicts:
            e_copy = dict(e)
            if e_copy.get("type") == "PERSON":
                text = e_copy["text"]
                pseudonym = self._get_person_pseudonym(text)
                e_copy["text"] = pseudonym
                e_copy["original_text"] = text
            anonymized.append(e_copy)
        return anonymized

    def _anonymize_kg_relationships(self, rel_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace PERSON source/target names with pseudonyms in KG relationship dicts.
        """
        anonymized = []
        for r in rel_dicts:
            r_copy = dict(r)
            for role in ("source", "target"):
                name = r_copy.get(role, "")
                role_type = r_copy.get(f"{role}_type", "")
                if role_type in ("PERSON", "person") and name:
                    r_copy[role] = self._get_person_pseudonym(name)
            anonymized.append(r_copy)
        return anonymized

    def _anonymize_participants(self, participants: List[str]) -> List[str]:
        """
        Anonymize a list of participant names using identity registry.
        """
        result = []
        for p in participants:
            pseudonym = None
            if self.identity_registry:
                pseudonym = self.identity_registry.get_pseudonym(p)
            if pseudonym:
                result.append(pseudonym)
            else:
                # Fall back to full anonymizer (handles emails, etc.)
                anon = self.anonymizer.anonymize(p, "en")
                result.append(anon.anonymized_text)
        return result

    def _extract_kg_relationships(
        self,
        text: str,
        entities: List[KGEntity],
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using modular extractor.

        Delegates to configured relationship extractor (cooccurrence, LLM, hybrid).
        Returns list of relationship dicts for PathRAG knowledge graph.
        """
        if not self.extract_relationships or not self.relationship_extractor:
            return []

        if len(entities) < 2:
            return []

        relationships = self.relationship_extractor.extract(text, entities, language)
        relationship_dicts = [r.to_dict() for r in relationships]
        self.stats["kg_relationships_extracted"] += len(relationship_dicts)
        return relationship_dicts

    def _save_attachment_chunk(self, chunk: ThreadChunk) -> None:
        """Save attachment chunk to Silver layer attachment_chunks/{classification}/"""
        subdir = chunk.attachment_classification or "unclassified"
        chunk_dir = self.silver_path / "attachment_chunks" / subdir
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f"{chunk.chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _process_attachments_separately(
        self,
        attachment_contents: list,
        thread_id: str,
        subject: str,
        participants: List[str],
        email_count: int,
        language: str,
    ) -> List[ThreadChunk]:
        """
        Process attachments separately from email body text.

        Each attachment is classified, optionally truncated, chunked,
        anonymized, and stored in attachment_chunks/.
        """
        chunks = []

        for att_content in attachment_contents:
            if not att_content.extraction_success or not att_content.text.strip():
                continue

            # Use classification from Bronze layer (set by AttachmentClassifier)
            classification = att_content.classification or "knowledge"

            # Only process knowledge attachments; skip transactional and other
            if classification != "knowledge":
                logger.info(f"Skipping {classification} attachment '{att_content.filename}'")
                self.stats["attachments_skipped_non_knowledge"] += 1
                continue

            self.stats["attachments_with_text"] += 1

            # Optionally truncate knowledge attachments
            text = att_content.text
            if self.max_tokens_knowledge > 0:
                max_chars = self.max_tokens_knowledge * 4
                if len(text) > max_chars:
                    text = text[:max_chars]
                    logger.info(
                        f"Truncated knowledge attachment '{att_content.filename}' "
                        f"from {len(att_content.text)} to {max_chars} chars"
                    )

            # Chunk the attachment text
            text_chunks = self.chunker.chunk(
                text=text,
                doc_id=att_content.attachment_id,
                metadata={"filename": att_content.filename},
            )

            for chunk in text_chunks:
                # Anonymize
                anon_result = self.anonymizer.anonymize(chunk.text, language)
                self.stats["pii_detected"] += anon_result.entity_count

                # Extract KG entities
                kg_entity_dicts, kg_entities_raw = self._extract_kg_entities(chunk.text, language)

                # Extract relationships
                kg_relationships = self._extract_kg_relationships(chunk.text, kg_entities_raw, language)

                # Anonymize KG metadata
                anon_kg_entities = self._anonymize_kg_entities(kg_entity_dicts)
                anon_kg_rels = self._anonymize_kg_relationships(kg_relationships)
                anon_participants = self._anonymize_participants(participants[:10])

                thread_chunk = ThreadChunk(
                    chunk_id=f"att_{att_content.attachment_id}_{chunk.chunk_index}",
                    thread_id=thread_id,
                    chunk_index=chunk.chunk_index,
                    text_original=chunk.text,
                    text_anonymized=anon_result.anonymized_text,
                    token_count=chunk.token_count,
                    thread_subject=subject,
                    thread_participants=anon_participants,
                    thread_email_count=email_count,
                    email_position="attachment",
                    pii_entities=[e.to_dict() for e in anon_result.entities],
                    pii_count=anon_result.entity_count,
                    kg_entities=anon_kg_entities,
                    kg_relationships=anon_kg_rels,
                    has_attachments=True,
                    attachment_count=1,
                    attachment_filenames=[att_content.filename],
                    source_type="attachment",
                    source_attachment_filename=att_content.filename,
                    attachment_classification=classification,
                    classification_confidence=getattr(att_content, "classification_confidence", 0.0),
                    language=language,
                    processing_mode=self.processing_mode,
                )

                chunks.append(thread_chunk)
                self._save_attachment_chunk(thread_chunk)
                self.stats["attachment_chunks_created"] += 1
                self.stats["chunks_created"] += 1

        return chunks

    def _create_directories(self) -> None:
        """Create Silver layer directory structure"""
        directories = [
            self.silver_path / "thread_chunks",                       # Thread-aware chunks
            self.silver_path / "thread_summaries",                    # Thread summaries
            self.silver_path / "individual_chunks",                   # Single email chunks
            self.silver_path / "attachment_chunks" / "knowledge",     # Knowledge attachments only
            self.silver_path / "pii_mappings",
            self.silver_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, int]:
        """
        Process Bronze layer emails into thread-aware Silver layer.

        Args:
            progress_callback: Optional callback(count, message)

        Returns:
            Processing statistics
        """
        self.stats["start_time"] = datetime.now().isoformat()

        # Step 1: Group emails into threads
        logger.info("Step 1: Grouping emails into threads...")
        threads = self.thread_grouper.group_from_bronze(str(self.bronze_path))

        if not threads:
            logger.warning("No emails found to process")
            return self.stats

        logger.info(f"Found {len(threads)} threads")

        # Step 2: Process each thread
        logger.info("Step 2: Processing threads...")

        for i, thread in enumerate(threads):
            try:
                if thread.is_thread:
                    # Multi-email thread: concatenate and chunk
                    self._process_thread(thread)
                    self.stats["multi_email_threads"] += 1
                else:
                    # Single email: process individually
                    self._process_single_email(thread)
                    self.stats["single_emails"] += 1

                self.stats["threads_processed"] += 1

                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, f"Processed {i + 1}/{len(threads)} threads")

            except Exception as e:
                logger.error(f"Error processing thread {thread.conversation_id}: {e}")
                self.stats["errors"] += 1

        # Step 3: Save metadata
        self.stats["end_time"] = datetime.now().isoformat()
        self._save_metadata()

        logger.info(f"Processing complete: {self.stats}")
        return self.stats

    def _process_thread(self, thread: EmailThread) -> List[ThreadChunk]:
        """Process a multi-email thread (attachments processed separately)"""
        chunks = []

        # Concatenate thread emails (body text only, no attachments)
        thread_text = thread.to_concatenated_text(include_metadata=True)
        thread_text = clean_email_text(thread_text)

        # Collect attachment metadata and contents for separate processing
        all_attachment_filenames = []
        total_attachments = 0
        all_attachment_contents = []

        if self.process_attachments and self.attachment_processor:
            for email in thread.emails:
                email_id = email.get('message_id', '')
                has_att = email.get('has_attachments', False)

                if email_id and has_att:
                    attachment_contents = self.attachment_processor.get_email_attachment_content(email_id)

                    for att_content in attachment_contents:
                        self.stats["attachments_processed"] += 1
                        all_attachment_filenames.append(att_content.filename)
                        total_attachments += 1
                        all_attachment_contents.append(att_content)

        if not thread_text.strip():
            return chunks

        has_attachments = total_attachments > 0

        # Detect language
        lang_result = self.language_detector.detect(thread_text)
        language = lang_result.language

        # Chunk the email body text (no attachment content)
        text_chunks = self.chunker.chunk(
            text=thread_text,
            doc_id=thread.conversation_id,
            metadata={
                "thread_subject": thread.subject,
                "thread_participants": thread.participants,
            }
        )

        # Process each email body chunk
        for chunk in text_chunks:
            # Anonymize
            anon_result = self.anonymizer.anonymize(chunk.text, language)
            self.stats["pii_detected"] += anon_result.entity_count

            # Extract KG entities (for PathRAG)
            kg_entity_dicts, kg_entities_raw = self._extract_kg_entities(chunk.text, language)

            # Extract relationships between entities (for PathRAG)
            kg_relationships = self._extract_kg_relationships(chunk.text, kg_entities_raw, language)

            # Anonymize KG metadata (entities, relationships, participants)
            anon_kg_entities = self._anonymize_kg_entities(kg_entity_dicts)
            anon_kg_rels = self._anonymize_kg_relationships(kg_relationships)
            anon_participants = self._anonymize_participants(thread.participants[:10])

            # Create thread chunk (source_type="email")
            thread_chunk = ThreadChunk(
                chunk_id=f"{thread.conversation_id}_{chunk.chunk_index}",
                thread_id=thread.conversation_id,
                chunk_index=chunk.chunk_index,
                text_original=chunk.text,
                text_anonymized=anon_result.anonymized_text,
                token_count=chunk.token_count,
                thread_subject=thread.subject,
                thread_participants=anon_participants,
                thread_email_count=thread.email_count,
                email_position=f"thread_{thread.email_count}_emails",
                pii_entities=[e.to_dict() for e in anon_result.entities],
                pii_count=anon_result.entity_count,
                kg_entities=anon_kg_entities,
                kg_relationships=anon_kg_rels,
                has_attachments=has_attachments,
                attachment_count=total_attachments,
                attachment_filenames=all_attachment_filenames,
                source_type="email",
                language=language,
                processing_mode=self.processing_mode,
            )

            chunks.append(thread_chunk)
            self._save_thread_chunk(thread_chunk)
            self.stats["chunks_created"] += 1

        # Process attachments separately → attachment_chunks/
        if all_attachment_contents:
            att_chunks = self._process_attachments_separately(
                attachment_contents=all_attachment_contents,
                thread_id=thread.conversation_id,
                subject=thread.subject,
                participants=thread.participants,
                email_count=thread.email_count,
                language=language,
            )
            chunks.extend(att_chunks)

        # Generate thread summary (email body chunks only)
        email_chunks = [c for c in chunks if c.source_type == "email"]
        if self.generate_summaries and email_chunks:
            self._generate_thread_summary(thread, email_chunks, language)

        return chunks

    def _process_single_email(self, thread: EmailThread) -> List[ThreadChunk]:
        """Process a single email (attachments processed separately)"""
        chunks = []
        email = thread.emails[0] if thread.emails else None

        if not email:
            return chunks

        # Get email body text only (no attachments)
        email_text, _ = self._format_single_email(email, include_attachments=False)
        email_text = clean_email_text(email_text)

        # Collect attachment contents separately
        attachment_filenames = []
        attachment_contents = []

        if self.process_attachments and self.attachment_processor and email.get('has_attachments'):
            email_id = email.get('message_id', '')
            if email_id:
                raw_contents = self.attachment_processor.get_email_attachment_content(email_id)
                for att_content in raw_contents:
                    self.stats["attachments_processed"] += 1
                    attachment_filenames.append(att_content.filename)
                    attachment_contents.append(att_content)

        if not email_text.strip():
            return chunks

        has_attachments = len(attachment_filenames) > 0 or email.get('has_attachments', False)
        attachment_count = len(attachment_filenames) or email.get('attachment_count', 0)

        # Detect language
        lang_result = self.language_detector.detect(email_text)
        language = lang_result.language

        # Chunk the email body text
        text_chunks = self.chunker.chunk(
            text=email_text,
            doc_id=email.get('message_id', thread.conversation_id),
            metadata={"subject": thread.subject}
        )

        # Process each email body chunk
        for chunk in text_chunks:
            # Anonymize
            anon_result = self.anonymizer.anonymize(chunk.text, language)
            self.stats["pii_detected"] += anon_result.entity_count

            # Extract KG entities (for PathRAG)
            kg_entity_dicts, kg_entities_raw = self._extract_kg_entities(chunk.text, language)

            # Extract relationships between entities (for PathRAG)
            kg_relationships = self._extract_kg_relationships(chunk.text, kg_entities_raw, language)

            # Anonymize KG metadata
            anon_kg_entities = self._anonymize_kg_entities(kg_entity_dicts)
            anon_kg_rels = self._anonymize_kg_relationships(kg_relationships)
            anon_participants = self._anonymize_participants(thread.participants[:10])

            # Create chunk (source_type="email", stored in individual_chunks)
            thread_chunk = ThreadChunk(
                chunk_id=f"{email.get('message_id', 'unknown')}_{chunk.chunk_index}",
                thread_id=thread.conversation_id,
                chunk_index=chunk.chunk_index,
                text_original=chunk.text,
                text_anonymized=anon_result.anonymized_text,
                token_count=chunk.token_count,
                thread_subject=thread.subject,
                thread_participants=anon_participants,
                thread_email_count=1,
                email_position="1/1",
                pii_entities=[e.to_dict() for e in anon_result.entities],
                pii_count=anon_result.entity_count,
                kg_entities=anon_kg_entities,
                kg_relationships=anon_kg_rels,
                has_attachments=has_attachments,
                attachment_count=attachment_count,
                attachment_filenames=attachment_filenames,
                source_type="email",
                language=language,
                processing_mode=self.processing_mode,
            )

            chunks.append(thread_chunk)
            self._save_individual_chunk(thread_chunk)
            self.stats["chunks_created"] += 1

        # Process attachments separately → attachment_chunks/
        if attachment_contents:
            att_chunks = self._process_attachments_separately(
                attachment_contents=attachment_contents,
                thread_id=thread.conversation_id,
                subject=thread.subject,
                participants=thread.participants,
                email_count=1,
                language=language,
            )
            chunks.extend(att_chunks)

        return chunks

    def _format_single_email(
        self,
        email: Dict[str, Any],
        include_attachments: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Format a single email for processing, including attachment content.

        Args:
            email: Email data dictionary
            include_attachments: Whether to include attachment text

        Returns:
            Tuple of (formatted_text, attachment_filenames)
        """
        parts = []
        attachment_filenames = []

        if email.get('subject'):
            parts.append(f"Subject: {email['subject']}")
        if email.get('sender'):
            parts.append(f"From: {email['sender']}")
        if email.get('sent_time'):
            parts.append(f"Date: {email['sent_time']}")

        parts.append("")

        if email.get('body_text'):
            parts.append(email['body_text'])

        # Process attachments if enabled
        if (include_attachments and self.include_attachment_text and
            self.attachment_processor and email.get('has_attachments')):

            email_id = email.get('message_id', '')
            if email_id:
                attachment_contents = self.attachment_processor.get_email_attachment_content(email_id)

                for att_content in attachment_contents:
                    self.stats["attachments_processed"] += 1
                    attachment_filenames.append(att_content.filename)

                    if att_content.extraction_success and att_content.text.strip():
                        self.stats["attachments_with_text"] += 1
                        parts.append("")
                        parts.append(f"--- Attachment: {att_content.filename} ---")
                        parts.append(att_content.text)
                        parts.append("--- End Attachment ---")

        return "\n".join(parts), attachment_filenames

    def _generate_thread_summary(
        self,
        thread: EmailThread,
        chunks: List[ThreadChunk],
        language: str
    ) -> Optional[ThreadSummary]:
        """Generate a summary for the thread"""
        # Build date range string
        date_range = ""
        if thread.start_date and thread.end_date:
            if thread.start_date.date() == thread.end_date.date():
                date_range = thread.start_date.strftime("%Y-%m-%d")
            else:
                date_range = f"{thread.start_date.strftime('%Y-%m-%d')} to {thread.end_date.strftime('%Y-%m-%d')}"

        # Extract key topics (simple: from subject + common words)
        key_topics = self._extract_key_topics(thread, chunks)

        # Generate summary
        if self.openai_api_key:
            summary_text = self._generate_llm_summary(thread, chunks)
        else:
            summary_text = self._generate_simple_summary(thread, chunks)

        summary = ThreadSummary(
            thread_id=thread.conversation_id,
            subject=thread.subject,
            participants=[self._anonymize_participant(p) for p in thread.participants[:10]],
            email_count=thread.email_count,
            date_range=date_range,
            summary=summary_text,
            key_topics=key_topics,
            chunk_ids=[c.chunk_id for c in chunks],
        )

        self._save_thread_summary(summary)
        self.stats["summaries_generated"] += 1

        return summary

    def _extract_key_topics(
        self,
        thread: EmailThread,
        chunks: List[ThreadChunk]
    ) -> List[str]:
        """Extract key topics from thread"""
        # Simple extraction: words from subject
        topics = []
        subject_words = thread.subject.lower().split()
        stopwords = {'re', 'fw', 'fwd', 'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were'}

        for word in subject_words:
            word = word.strip(':-.,!?()[]')
            if len(word) > 2 and word not in stopwords:
                topics.append(word)

        return topics[:5]

    def _generate_simple_summary(
        self,
        thread: EmailThread,
        chunks: List[ThreadChunk]
    ) -> str:
        """Generate a simple summary without LLM"""
        # Use anonymized text from first and last chunk
        if not chunks:
            return "No content available"

        first_chunk = chunks[0].text_anonymized[:200]
        summary = f"Thread with {thread.email_count} emails about '{thread.subject}'. "
        summary += f"Participants: {', '.join(thread.participants[:3])}. "
        summary += f"Preview: {first_chunk}..."

        return summary

    def _generate_llm_summary(
        self,
        thread: EmailThread,
        chunks: List[ThreadChunk]
    ) -> str:
        """Generate summary using LLM (supports both OpenAI and Azure OpenAI)"""
        try:
            if self.use_azure:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                )
                model = self.azure_deployment or "gpt-4o"
            else:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                model = "gpt-4o-mini"

            # Combine anonymized chunk texts
            combined_text = "\n\n".join([c.text_anonymized for c in chunks])[:4000]

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this email thread in 2-3 sentences. Focus on the main topic and outcome."
                    },
                    {
                        "role": "user",
                        "content": combined_text
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return self._generate_simple_summary(thread, chunks)

    def _anonymize_participant(self, participant: str) -> str:
        """Anonymize participant name"""
        # Use the same anonymizer for consistency
        result = self.anonymizer.anonymize(participant, "en")
        return result.anonymized_text

    def _save_thread_chunk(self, chunk: ThreadChunk) -> None:
        """Save thread chunk to Silver layer"""
        chunk_file = self.silver_path / "thread_chunks" / f"{chunk.chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_individual_chunk(self, chunk: ThreadChunk) -> None:
        """Save individual email chunk to Silver layer"""
        chunk_file = self.silver_path / "individual_chunks" / f"{chunk.chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_thread_summary(self, summary: ThreadSummary) -> None:
        """Save thread summary to Silver layer"""
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary.thread_id)[:100]
        summary_file = self.silver_path / "thread_summaries" / f"{safe_id}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_metadata(self) -> None:
        """Save processing metadata"""
        metadata_file = self.silver_path / "metadata" / "thread_processing_stats.json"

        # Load existing
        existing = []
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(self.stats)

        with open(metadata_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        # Save PII mapping
        mapping_file = self.silver_path / "pii_mappings" / "thread_mapping.json"
        mapping = self.anonymizer.get_mapping()
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)


# Convenience function
def process_threads_to_silver(
    bronze_path: str,
    silver_path: str,
    chunk_size: int = 512,
    openai_api_key: Optional[str] = None
) -> Dict[str, int]:
    """
    Process Bronze layer emails into thread-aware Silver layer.

    Args:
        bronze_path: Path to Bronze layer
        silver_path: Path for Silver layer output
        chunk_size: Target chunk size in tokens
        openai_api_key: Optional OpenAI key for summaries

    Returns:
        Processing statistics
    """
    processor = ThreadAwareProcessor(
        bronze_path=bronze_path,
        silver_path=silver_path,
        chunk_size=chunk_size,
        openai_api_key=openai_api_key
    )

    return processor.process()
