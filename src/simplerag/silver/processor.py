"""
Silver Layer - Text Extraction, OCR, Anonymization & Summarization

Objective: Create clean, anonymized, human-readable representations of all content.

Input: Read only from bronze_data
Output: Structured JSON to silver_data

Processes:
1. Text-based content: Normalize, detect language, anonymize PII, summarize
2. Images/Image-based PDFs: OCR via Azure Vision, normalize, anonymize, summarize

All AI operations record model name, version, and prompt version for lineage.

Output template supports: SimpleRAG, PathRAG, GraphRAG
"""

import json
import asyncio
import logging
import base64
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from openai import AsyncAzureOpenAI
import httpx

from ..bronze.ingestion import BronzeRecord
from ..utils.lineage import LineageTracker, LineageRecord
from ..utils.config import SimpleRAGConfig
from .disclaimer_remover import remove_disclaimers_with_info

logger = logging.getLogger(__name__)


# =============================================================================
# Silver Record Structure - Supports SimpleRAG, PathRAG, GraphRAG
# =============================================================================

@dataclass
class Participant:
    """Email participant with structured fields for GraphRAG."""
    email: str = ""
    name: str = ""
    domain: str = ""

    @staticmethod
    def parse(raw: str) -> 'Participant':
        """Parse 'Name <email@domain>' or 'email@domain' format."""
        if not raw:
            return Participant()

        raw = raw.strip()

        # Pattern 1: "Name" <email@domain> or Name <email@domain>
        match = re.match(r'^["\']?([^"\'<]+)["\']?\s*<([^<>]+@([^<>]+))>$', raw)
        if match:
            name = match.group(1).strip().strip('"\'')
            email = match.group(2).strip()
            domain = match.group(3).strip()
            return Participant(email=email, name=name, domain=domain)

        # Pattern 2: Just email@domain
        if '@' in raw and '<' not in raw:
            email = raw.strip()
            domain = email.split('@')[-1] if '@' in email else ""
            return Participant(email=email, name="", domain=domain)

        # Pattern 3: Just a name
        return Participant(name=raw.strip())


@dataclass
class Participants:
    """All participants in an email - for GraphRAG relationship building."""
    sender: Participant = field(default_factory=Participant)
    to: List[Participant] = field(default_factory=list)
    cc: List[Participant] = field(default_factory=list)
    bcc: List[Participant] = field(default_factory=list)


@dataclass
class Metadata:
    """Email metadata."""
    subject: str = ""
    timestamp: str = ""  # ISO-8601
    language: str = "en"
    source_type: str = "pst"  # pst, imap, exchange, gmail, file
    processing_time: str = ""  # ISO-8601


@dataclass
class Content:
    """Email content - raw and processed."""
    raw_text: str = ""
    clean_text: str = ""  # Anonymized, disclaimer-removed
    summary: str = ""
    word_count: int = 0
    has_attachments: bool = False
    attachment_count: int = 0


@dataclass
class Anonymization:
    """PII anonymization details."""
    pii_mappings: Dict[str, str] = field(default_factory=dict)
    version: str = "v1.0"


@dataclass
class Threading:
    """Email threading info - for PathRAG conversation tracing."""
    message_id: str = ""
    in_reply_to: str = ""
    references: List[str] = field(default_factory=list)
    is_reply: bool = False
    thread_position: int = 0  # Position in thread (0 = root)


@dataclass
class Quality:
    """Processing quality indicators."""
    disclaimer_removed: bool = False
    signature_removed: bool = False
    ocr_performed: bool = False
    cleaning_version: str = "v1.0"


@dataclass
class Chunk:
    """Text chunk for RAG retrieval."""
    chunk_id: str = ""
    text: str = ""
    word_count: int = 0


@dataclass
class Lineage:
    """Data lineage tracking."""
    bronze_ref: str = ""
    source_file: str = ""
    attachment_refs: List[str] = field(default_factory=list)


@dataclass
class SilverRecord:
    """
    Silver layer record - structured for SimpleRAG, PathRAG, GraphRAG.

    All fields support downstream RAG approaches:
    - SimpleRAG: content.clean_text, chunks
    - PathRAG: threading (message_id, in_reply_to, references)
    - GraphRAG: participants (sender, to, cc with email/name/domain)
    """
    record_id: str
    email_id: str  # Bronze record ID
    thread_id: str = ""  # Thread grouping ID

    metadata: Metadata = field(default_factory=Metadata)
    participants: Participants = field(default_factory=Participants)
    content: Content = field(default_factory=Content)
    anonymization: Anonymization = field(default_factory=Anonymization)
    threading: Threading = field(default_factory=Threading)
    quality: Quality = field(default_factory=Quality)
    chunks: List[Chunk] = field(default_factory=list)
    lineage: Lineage = field(default_factory=Lineage)

    def to_dict(self) -> Dict:
        """Convert to dictionary with nested structures."""
        return {
            "record_id": self.record_id,
            "email_id": self.email_id,
            "thread_id": self.thread_id,
            "metadata": asdict(self.metadata),
            "participants": {
                "from": asdict(self.participants.sender),
                "to": [asdict(p) for p in self.participants.to],
                "cc": [asdict(p) for p in self.participants.cc],
                "bcc": [asdict(p) for p in self.participants.bcc],
            },
            "content": asdict(self.content),
            "anonymization": {
                "pii_mappings": self.anonymization.pii_mappings,
                "version": self.anonymization.version,
            },
            "threading": {
                "message_id": self.threading.message_id,
                "in_reply_to": self.threading.in_reply_to,
                "references": self.threading.references,
                "is_reply": self.threading.is_reply,
                "thread_position": self.threading.thread_position,
            },
            "quality": asdict(self.quality),
            "chunks": [asdict(c) for c in self.chunks],
            "lineage": asdict(self.lineage),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SilverRecord':
        """Load from dictionary."""
        record = cls(
            record_id=data["record_id"],
            email_id=data["email_id"],
            thread_id=data.get("thread_id", ""),
        )

        # Metadata
        if "metadata" in data:
            record.metadata = Metadata(**data["metadata"])

        # Participants
        if "participants" in data:
            p = data["participants"]
            record.participants = Participants(
                sender=Participant(**p.get("from", {})),
                to=[Participant(**x) for x in p.get("to", [])],
                cc=[Participant(**x) for x in p.get("cc", [])],
                bcc=[Participant(**x) for x in p.get("bcc", [])],
            )

        # Content
        if "content" in data:
            record.content = Content(**data["content"])

        # Anonymization
        if "anonymization" in data:
            record.anonymization = Anonymization(**data["anonymization"])

        # Threading
        if "threading" in data:
            record.threading = Threading(**data["threading"])

        # Quality
        if "quality" in data:
            record.quality = Quality(**data["quality"])

        # Chunks
        if "chunks" in data:
            record.chunks = [Chunk(**c) for c in data["chunks"]]

        # Lineage
        if "lineage" in data:
            record.lineage = Lineage(**data["lineage"])

        return record


# =============================================================================
# Silver Processor
# =============================================================================

class SilverProcessor:
    """
    Silver Layer Processor - Enrich Bronze data with AI.

    Process:
    1. Read from Bronze layer ONLY
    2. Apply OCR if needed (Azure Vision)
    3. Anonymize PII (Azure OpenAI)
    4. Generate summaries (Azure OpenAI)
    5. Chunk for RAG
    6. Store with full lineage tracking
    """

    # Prompt versions for lineage tracking
    PROMPT_VERSION_ANONYMIZE = "v1.0"
    PROMPT_VERSION_SUMMARIZE = "v1.0"
    PROMPT_VERSION_OCR = "v1.0"
    CLEANING_VERSION = "v1.0"

    def __init__(self, config: SimpleRAGConfig, lineage_tracker: LineageTracker):
        self.config = config
        self.lineage = lineage_tracker
        self.dirs = config.directories

        # Azure OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version
        )

        # HTTP client for Azure Vision OCR
        self.http_client = httpx.AsyncClient(timeout=60.0)

        logger.info("Silver processor initialized")

    async def process_bronze_record(self, bronze_record: BronzeRecord) -> Optional[SilverRecord]:
        """
        Process a single Bronze record through Silver layer.

        Returns SilverRecord with full lineage tracking.
        """
        logger.info(f"Processing Bronze record: {bronze_record.record_id}")

        # Create lineage tracking record
        lineage_record = self.lineage.create_record(
            record_id=f"silver_{bronze_record.record_id}",
            bronze_source_path=str(self.config.directories.bronze_dir / "emails" / f"{bronze_record.record_id}.json"),
            original_filename=bronze_record.source_file,
            file_type=bronze_record.file_type
        )

        try:
            # Step 1: Extract text (with OCR if needed)
            raw_text, ocr_performed = await self._extract_text(bronze_record, lineage_record)

            if not raw_text.strip():
                logger.warning(f"No text extracted from {bronze_record.record_id}")
                return None

            # Step 2: Clean text (remove disclaimers)
            clean_text, removed_info = remove_disclaimers_with_info(raw_text)
            disclaimer_removed = bool(removed_info)

            # Step 3: Detect language
            detected_language = self._detect_language(clean_text)

            # Step 4: Anonymize PII
            anonymized_text, pii_mappings = await self._anonymize_text(clean_text, lineage_record)

            # Step 5: Generate summary
            summary = await self._summarize_text(anonymized_text, lineage_record)

            # Step 6: Chunk for RAG
            chunks = self._chunk_text(anonymized_text, bronze_record.record_id)

            # Build structured Silver record
            silver_record = self._build_silver_record(
                bronze_record=bronze_record,
                raw_text=raw_text,
                anonymized_text=anonymized_text,
                summary=summary,
                pii_mappings=pii_mappings,
                chunks=chunks,
                detected_language=detected_language,
                ocr_performed=ocr_performed,
                disclaimer_removed=disclaimer_removed,
            )

            # Save Silver record
            self._save_silver_record(silver_record)

            # Update lineage
            lineage_record.silver_output_path = str(self.dirs.silver_dir / "processed" / f"{silver_record.record_id}.json")
            lineage_record.silver_processing_time = datetime.utcnow().isoformat()
            self.lineage.save_record(lineage_record)

            logger.info(f"Silver processing complete: {silver_record.record_id}")
            return silver_record

        except Exception as e:
            logger.error(f"Silver processing failed for {bronze_record.record_id}: {e}")
            raise

    def _build_silver_record(
        self,
        bronze_record: BronzeRecord,
        raw_text: str,
        anonymized_text: str,
        summary: str,
        pii_mappings: Dict[str, str],
        chunks: List[Dict],
        detected_language: str,
        ocr_performed: bool,
        disclaimer_removed: bool,
    ) -> SilverRecord:
        """Build structured SilverRecord from Bronze data and processing results."""

        metadata = bronze_record.document_metadata or {}
        headers = bronze_record.email_headers or {}

        # Parse participants for GraphRAG
        participants = Participants(
            sender=Participant.parse(metadata.get("sender_name", "") or metadata.get("sender_email", "")),
            to=self._parse_recipients(metadata.get("recipients", "")),
            cc=[],  # Not in current Bronze structure
            bcc=[],
        )

        # Threading info for PathRAG
        in_reply_to = headers.get("in_reply_to", "")
        references_raw = headers.get("references", "")
        references = [r.strip() for r in references_raw.split() if r.strip()] if references_raw else []

        threading = Threading(
            message_id=headers.get("message_id", ""),
            in_reply_to=in_reply_to,
            references=references,
            is_reply=bool(in_reply_to),
            thread_position=len(references) if references else (1 if in_reply_to else 0),
        )

        # Build record
        return SilverRecord(
            record_id=f"silver_{bronze_record.record_id}",
            email_id=bronze_record.record_id,
            thread_id=bronze_record.email_thread_id or "",

            metadata=Metadata(
                subject=metadata.get("subject", ""),
                timestamp=metadata.get("sent_time", ""),
                language=detected_language,
                source_type=bronze_record.file_type.lstrip("."),
                processing_time=datetime.utcnow().isoformat(),
            ),

            participants=participants,

            content=Content(
                raw_text=raw_text,
                clean_text=anonymized_text,
                summary=summary,
                word_count=len(anonymized_text.split()),
                has_attachments=len(bronze_record.email_attachments) > 0,
                attachment_count=len(bronze_record.email_attachments),
            ),

            anonymization=Anonymization(
                pii_mappings=pii_mappings,
                version=self.PROMPT_VERSION_ANONYMIZE,
            ),

            threading=threading,

            quality=Quality(
                disclaimer_removed=disclaimer_removed,
                signature_removed=disclaimer_removed,  # Combined in disclaimer removal
                ocr_performed=ocr_performed,
                cleaning_version=self.CLEANING_VERSION,
            ),

            chunks=[
                Chunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    word_count=c["word_count"],
                )
                for c in chunks
            ],

            lineage=Lineage(
                bronze_ref=bronze_record.record_id,
                source_file=bronze_record.source_file,
                attachment_refs=[a.get("attachment_id", "") for a in bronze_record.email_attachments],
            ),
        )

    def _parse_recipients(self, recipients_str: str) -> List[Participant]:
        """Parse comma/semicolon separated recipients."""
        if not recipients_str:
            return []

        # Split by comma or semicolon
        parts = re.split(r'[,;]', recipients_str)
        return [Participant.parse(p.strip()) for p in parts if p.strip()]

    async def _extract_text(
        self,
        bronze_record: BronzeRecord,
        lineage: LineageRecord
    ) -> Tuple[str, bool]:
        """Extract text from Bronze record."""
        ocr_performed = False

        # Check if OCR is needed
        needs_ocr = (
            bronze_record.document_metadata and
            bronze_record.document_metadata.get("requires_ocr", False)
        )

        if needs_ocr and bronze_record.raw_binary_path:
            text = await self._perform_ocr(bronze_record.raw_binary_path, lineage)
            return text, True

        # Text-based content
        if bronze_record.raw_content:
            return bronze_record.raw_content, False

        # Email content
        if bronze_record.email_body_text:
            return bronze_record.email_body_text, False

        # Fallback for HTML only
        if bronze_record.email_body_html:
            text = re.sub(r'<[^>]+>', ' ', bronze_record.email_body_html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text, False

        return "", False

    async def _perform_ocr(self, image_path: str, lineage: LineageRecord) -> str:
        """Perform OCR using Azure Vision API."""
        if not self.config.azure_vision.endpoint:
            return await self._perform_ocr_openai(image_path, lineage)

        logger.info(f"Performing Azure Vision OCR on: {image_path}")

        with open(image_path, 'rb') as f:
            image_data = f.read()

        url = f"{self.config.azure_vision.endpoint}/computervision/imageanalysis:analyze"
        params = {"api-version": self.config.azure_vision.api_version, "features": "read"}
        headers = {
            "Ocp-Apim-Subscription-Key": self.config.azure_vision.api_key,
            "Content-Type": "application/octet-stream"
        }

        response = await self.http_client.post(url, params=params, headers=headers, content=image_data)
        response.raise_for_status()
        result = response.json()

        text_lines = []
        if "readResult" in result:
            for block in result["readResult"].get("blocks", []):
                for line in block.get("lines", []):
                    text_lines.append(line.get("text", ""))

        extracted_text = "\n".join(text_lines)

        lineage.add_ai_operation(
            operation_type="ocr",
            model_name=self.config.azure_vision.model_name,
            model_version=self.config.azure_vision.model_version,
            prompt_version=self.PROMPT_VERSION_OCR,
            input_data=image_path,
            output_data=extracted_text[:500]
        )

        return extracted_text

    async def _perform_ocr_openai(self, image_path: str, lineage: LineageRecord) -> str:
        """Perform OCR using Azure OpenAI GPT-4o Vision."""
        logger.info(f"Performing GPT-4o Vision OCR on: {image_path}")

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        suffix = Path(image_path).suffix.lower()
        mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
        mime_type = mime_map.get(suffix, 'image/png')

        response = await self.openai_client.chat.completions.create(
            model=self.config.azure_openai.deployment,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image. Return only the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                ]
            }],
            max_tokens=4000
        )

        extracted_text = response.choices[0].message.content or ""

        lineage.add_ai_operation(
            operation_type="ocr",
            model_name=self.config.azure_openai.model_name,
            model_version=self.config.azure_openai.model_version,
            prompt_version=self.PROMPT_VERSION_OCR,
            input_data=image_path,
            output_data=extracted_text[:500]
        )

        return extracted_text

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        dutch_words = {'de', 'het', 'een', 'en', 'van', 'in', 'is', 'dat', 'niet', 'te'}
        words = set(text.lower().split()[:100])
        if len(words & dutch_words) > 5:
            return "nl"
        return "en"

    async def _anonymize_text(
        self,
        text: str,
        lineage: LineageRecord
    ) -> Tuple[str, Dict[str, str]]:
        """Anonymize PII using Azure OpenAI."""
        logger.info("Anonymizing PII...")

        prompt = """Anonymize the following text by replacing Personal Identifiable Information (PII) with placeholders.

Replace:
- Names → [PERSON_1], [PERSON_2], etc.
- Email addresses → [EMAIL_1], [EMAIL_2], etc.
- Phone numbers → [PHONE_1], [PHONE_2], etc.
- Addresses → [ADDRESS_1], etc.
- ID numbers → [ID_1], etc.
- Company names → [ORG_1], [ORG_2], etc. (if they could identify specific individuals)

Return a JSON object with:
1. "anonymized_text": the text with PII replaced
2. "mappings": a dictionary mapping placeholders to original values

Text to anonymize:
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.azure_openai.deployment,
                messages=[
                    {"role": "system", "content": "You are a PII anonymization assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt + text[:8000]}
                ],
                max_tokens=4000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content or "{}")
            anonymized_text = result.get("anonymized_text", text)
            pii_mappings = result.get("mappings", {})

            lineage.add_ai_operation(
                operation_type="anonymization",
                model_name=self.config.azure_openai.model_name,
                model_version=self.config.azure_openai.model_version,
                prompt_version=self.PROMPT_VERSION_ANONYMIZE,
                input_data=text[:500],
                output_data=anonymized_text[:500],
                parameters={"pii_count": len(pii_mappings)}
            )

            return anonymized_text, pii_mappings

        except Exception as e:
            logger.warning(f"Anonymization failed: {e}, returning original text")
            return text, {}

    async def _summarize_text(self, text: str, lineage: LineageRecord) -> str:
        """Generate summary using Azure OpenAI."""
        logger.info("Generating summary...")

        prompt = """Provide a concise summary of the following text.
The summary should:
- Be 2-3 sentences
- Capture the main points
- Be suitable for search/retrieval purposes

Text:
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.azure_openai.deployment,
                messages=[
                    {"role": "system", "content": "You are a summarization assistant. Be concise and factual."},
                    {"role": "user", "content": prompt + text[:6000]}
                ],
                max_tokens=500,
                temperature=0.3
            )

            summary = response.choices[0].message.content or ""

            lineage.add_ai_operation(
                operation_type="summarization",
                model_name=self.config.azure_openai.model_name,
                model_version=self.config.azure_openai.model_version,
                prompt_version=self.PROMPT_VERSION_SUMMARIZE,
                input_data=text[:500],
                output_data=summary
            )

            return summary.strip()

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return ""

    def _chunk_text(
        self,
        text: str,
        source_id: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict]:
        """Split text into chunks for RAG retrieval."""
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        chunks = []
        sentences = text.replace('\n', ' ').split('. ')

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = len(sentence.split())

            if current_length + sentence_words > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "chunk_id": f"{source_id}_chunk_{len(chunks)}",
                    "text": chunk_text,
                    "word_count": current_length,
                })

                # Overlap
                overlap_sentences = []
                overlap_words = 0
                for s in reversed(current_chunk):
                    if overlap_words + len(s.split()) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += len(s.split())
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_words

            current_chunk.append(sentence)
            current_length += sentence_words

        # Final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            chunks.append({
                "chunk_id": f"{source_id}_chunk_{len(chunks)}",
                "text": chunk_text,
                "word_count": current_length,
            })

        return chunks

    def _save_silver_record(self, record: SilverRecord):
        """Save Silver record to disk - single complete JSON file."""
        output_path = self.dirs.silver_dir / "processed" / f"{record.record_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    async def process_all_bronze(self, skip_existing: bool = True) -> List[SilverRecord]:
        """Process all Bronze records through Silver layer.

        Args:
            skip_existing: If True, skip Bronze records that already have Silver output
        """
        from ..bronze.ingestion import BronzeIngestion

        bronze = BronzeIngestion(self.config)
        records = []

        # Get already processed record IDs if skipping
        existing_ids = set()
        if skip_existing:
            processed_dir = self.dirs.silver_dir / "processed"
            if processed_dir.exists():
                for f in processed_dir.glob("*.json"):
                    # Extract bronze_id from silver_bronze_xxx.json
                    silver_id = f.stem  # silver_bronze_xxx
                    bronze_id = silver_id.replace("silver_", "", 1)  # bronze_xxx
                    existing_ids.add(bronze_id)
            if existing_ids:
                logger.info(f"Skipping {len(existing_ids)} already processed records")

        for bronze_record in bronze.list_bronze_records():
            # Skip if already processed
            if skip_existing and bronze_record.record_id in existing_ids:
                continue

            try:
                silver_record = await self.process_bronze_record(bronze_record)
                if silver_record:
                    records.append(silver_record)
            except Exception as e:
                logger.error(f"Failed to process {bronze_record.record_id}: {e}")

        return records

    def list_silver_records(self) -> List[SilverRecord]:
        """List all Silver records."""
        records = []
        processed_dir = self.dirs.silver_dir / "processed"

        if not processed_dir.exists():
            return records

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    records.append(SilverRecord.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return records

    def get_silver_record(self, record_id: str) -> Optional[SilverRecord]:
        """Get a specific Silver record."""
        path = self.dirs.silver_dir / "processed" / f"{record_id}.json"
        if path.exists():
            with open(path, 'r') as f:
                return SilverRecord.from_dict(json.load(f))
        return None

    async def close(self):
        """Close async resources."""
        await self.http_client.aclose()
