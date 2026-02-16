"""
Lineage Tracking for SimpleRAG Pipeline

Every output must preserve lineage to Bronze layer.
All AI operations must record model name, version, and prompt version.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class AIOperation:
    """Record of an AI operation for lineage tracking."""
    operation_type: str  # "ocr", "anonymization", "summarization", "embedding"
    model_name: str
    model_version: str
    prompt_version: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageRecord:
    """
    Complete lineage record for a processed item.

    Tracks the full transformation chain from Bronze to Gold.
    """
    # Identity
    record_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Bronze source (immutable reference)
    bronze_source_path: str = ""
    bronze_source_hash: str = ""
    bronze_ingestion_time: str = ""

    # Silver transformations
    silver_output_path: str = ""
    silver_processing_time: str = ""
    ai_operations: List[AIOperation] = field(default_factory=list)

    # Gold indexing
    gold_chunk_ids: List[str] = field(default_factory=list)
    gold_embedding_ids: List[str] = field(default_factory=list)
    gold_indexing_time: str = ""

    # Metadata
    file_type: str = ""
    original_filename: str = ""
    content_hash: str = ""

    def add_ai_operation(
        self,
        operation_type: str,
        model_name: str,
        model_version: str,
        prompt_version: str,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        """Add an AI operation to the lineage record."""
        op = AIOperation(
            operation_type=operation_type,
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            input_hash=self._compute_hash(input_data) if input_data else None,
            output_hash=self._compute_hash(output_data) if output_data else None,
            parameters=parameters or {}
        )
        self.ai_operations.append(op)
        return op

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert AIOperation objects
        data['ai_operations'] = [asdict(op) for op in self.ai_operations]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'LineageRecord':
        """Create from dictionary."""
        ai_ops = [AIOperation(**op) for op in data.pop('ai_operations', [])]
        record = cls(**data)
        record.ai_operations = ai_ops
        return record


class LineageTracker:
    """
    Manages lineage tracking across the entire pipeline.

    Ensures every output can be traced back to its Bronze source.
    """

    def __init__(self, lineage_dir: Path):
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.lineage_dir / "lineage_index.json"
        self._index: Dict[str, str] = {}  # record_id -> lineage_file_path
        self._load_index()

    def _load_index(self):
        """Load the lineage index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)

    def _save_index(self):
        """Save the lineage index."""
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def create_record(
        self,
        record_id: str,
        bronze_source_path: str,
        original_filename: str,
        file_type: str
    ) -> LineageRecord:
        """Create a new lineage record for a Bronze source."""
        # Compute Bronze source hash
        bronze_path = Path(bronze_source_path)
        bronze_hash = ""
        if bronze_path.exists():
            with open(bronze_path, 'rb') as f:
                bronze_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        record = LineageRecord(
            record_id=record_id,
            bronze_source_path=str(bronze_source_path),
            bronze_source_hash=bronze_hash,
            bronze_ingestion_time=datetime.utcnow().isoformat(),
            original_filename=original_filename,
            file_type=file_type
        )

        self.save_record(record)
        return record

    def save_record(self, record: LineageRecord):
        """Save a lineage record to disk."""
        record_path = self.lineage_dir / f"{record.record_id}.json"
        with open(record_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)

        self._index[record.record_id] = str(record_path)
        self._save_index()

        logger.debug(f"Saved lineage record: {record.record_id}")

    def get_record(self, record_id: str) -> Optional[LineageRecord]:
        """Retrieve a lineage record."""
        if record_id not in self._index:
            return None

        record_path = Path(self._index[record_id])
        if not record_path.exists():
            return None

        with open(record_path, 'r') as f:
            data = json.load(f)
            return LineageRecord.from_dict(data)

    def trace_to_bronze(self, record_id: str) -> Optional[Dict]:
        """
        Trace a record back to its Bronze source.

        Returns full lineage information including all AI operations.
        """
        record = self.get_record(record_id)
        if not record:
            return None

        return {
            "record_id": record.record_id,
            "bronze_source": {
                "path": record.bronze_source_path,
                "hash": record.bronze_source_hash,
                "ingestion_time": record.bronze_ingestion_time,
                "original_filename": record.original_filename,
                "file_type": record.file_type
            },
            "silver_transformations": {
                "output_path": record.silver_output_path,
                "processing_time": record.silver_processing_time,
                "ai_operations": [asdict(op) for op in record.ai_operations]
            },
            "gold_indexing": {
                "chunk_ids": record.gold_chunk_ids,
                "embedding_ids": record.gold_embedding_ids,
                "indexing_time": record.gold_indexing_time
            }
        }

    def verify_lineage_integrity(self, record_id: str) -> Dict[str, Any]:
        """
        Verify that a record's lineage is intact.

        Checks that Bronze source still exists and matches recorded hash.
        """
        record = self.get_record(record_id)
        if not record:
            return {"valid": False, "error": "Record not found"}

        bronze_path = Path(record.bronze_source_path)
        if not bronze_path.exists():
            return {
                "valid": False,
                "error": "Bronze source no longer exists",
                "bronze_path": str(bronze_path)
            }

        # Verify hash
        with open(bronze_path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        if current_hash != record.bronze_source_hash:
            return {
                "valid": False,
                "error": "Bronze source has been modified (hash mismatch)",
                "expected_hash": record.bronze_source_hash,
                "current_hash": current_hash
            }

        return {
            "valid": True,
            "record_id": record_id,
            "bronze_verified": True,
            "ai_operations_count": len(record.ai_operations)
        }

    def list_records(self) -> List[str]:
        """List all tracked record IDs."""
        return list(self._index.keys())

    def get_ai_operations_summary(self, record_id: str) -> Optional[List[Dict]]:
        """Get summary of all AI operations for a record."""
        record = self.get_record(record_id)
        if not record:
            return None

        return [
            {
                "operation": op.operation_type,
                "model": f"{op.model_name}@{op.model_version}",
                "prompt_version": op.prompt_version,
                "timestamp": op.timestamp
            }
            for op in record.ai_operations
        ]
