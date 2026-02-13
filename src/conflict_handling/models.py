"""
Data models for conflict handling system.

This module defines the core data structures used throughout the
conflict detection and resolution pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class PIIType(Enum):
    """Types of Personally Identifiable Information"""
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    ID_NUMBER = "ID_NUMBER"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    IP_ADDRESS = "IP_ADDRESS"
    LOCATION = "LOCATION"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicting information"""
    NEWER_WINS = "newer_wins"           # Most recent value wins
    OLDER_WINS = "older_wins"           # Original value preserved
    HIGHER_CONFIDENCE = "higher_confidence"  # Higher confidence wins
    AVERAGE = "average"                 # Average for numeric values
    USER_RESOLUTION = "user_resolution" # Prompt user to decide
    BOTH_VALID = "both_valid"           # Both values are valid (context-dependent)


class SourceType(Enum):
    """Types of document sources"""
    EMAIL = "email"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"
    HTML = "html"
    ATTACHMENT = "attachment"


# =============================================================================
# Bronze Layer Models
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Metadata extracted from documents at ingestion time.
    Used for temporal tracking and conflict resolution.
    """

    doc_id: str
    source_file: str
    source_type: SourceType

    # Temporal metadata
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    sent_date: Optional[datetime] = None          # For emails
    received_date: Optional[datetime] = None      # For emails
    extraction_date: datetime = field(default_factory=datetime.now)

    # Version metadata
    version: Optional[str] = None                 # "v2.1", "Draft 3"
    revision: Optional[int] = None                # Numeric revision
    supersedes: Optional[str] = None              # Reference to older doc
    superseded_by: Optional[str] = None           # Reference to newer doc

    # Author metadata
    author: Optional[str] = None
    last_modified_by: Optional[str] = None

    # Email-specific metadata
    email_subject: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    email_thread_id: Optional[str] = None

    # Quality indicators
    is_attachment: bool = False
    language: str = "en"
    confidence_score: float = 1.0

    def get_best_date(self) -> Optional[datetime]:
        """Get the most reliable date for this document"""
        # Priority: sent_date > modified_date > created_date > extraction_date
        return (
            self.sent_date or
            self.modified_date or
            self.created_date or
            self.extraction_date
        )


@dataclass
class TemporalSignals:
    """
    Temporal signals extracted from document content.
    Used when metadata is incomplete or unavailable.
    """

    # Explicit dates found in text
    explicit_dates: List[datetime] = field(default_factory=list)

    # Date expressions (raw text)
    date_expressions: List[str] = field(default_factory=list)

    # Relative time references
    relative_references: List[str] = field(default_factory=list)

    # Version indicators found in text
    version_indicators: List[str] = field(default_factory=list)

    # References to other documents
    supersedes_references: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)

    # Inferred date (best guess from content)
    inferred_date: Optional[datetime] = None

    # Confidence in temporal inference
    confidence: float = 0.0

    def has_signals(self) -> bool:
        """Check if any temporal signals were found"""
        return bool(
            self.explicit_dates or
            self.version_indicators or
            self.supersedes_references
        )


# =============================================================================
# Silver Layer Models
# =============================================================================

@dataclass
class PIIAnnotation:
    """Single PII annotation in text"""

    text: str                          # The actual PII text
    pii_type: PIIType                  # Type of PII
    start_char: int                    # Start position in text
    end_char: int                      # End position in text
    confidence: float                  # Detection confidence
    is_sensitive: bool = True          # Should be anonymized
    anonymized_text: Optional[str] = None  # Replacement text


@dataclass
class EntityMention:
    """Entity mention extracted from text"""

    text: str                          # Original text
    entity_type: str                   # PERSON, ORGANIZATION, PROJECT, etc.
    start_char: int
    end_char: int
    confidence: float
    normalized_name: Optional[str] = None  # Canonical form


@dataclass
class EnrichedChunk:
    """
    Enriched text chunk with all extracted information.
    This is the primary unit of data in the Silver layer.
    """

    # Identification
    chunk_id: str
    doc_id: str                        # Parent document
    chunk_index: int                   # Position in document

    # Content
    text: str
    text_anonymized: Optional[str] = None

    # Temporal information
    source_date: Optional[datetime] = None
    extraction_date: datetime = field(default_factory=datetime.now)
    temporal_signals: Optional[TemporalSignals] = None

    # Extracted entities
    entities: List[EntityMention] = field(default_factory=list)
    pii_annotations: List[PIIAnnotation] = field(default_factory=list)

    # NLP results
    language: str = "en"
    summary: Optional[str] = None

    # Embedding (added in Gold layer)
    embedding: Optional[List[float]] = None

    # Quality metrics
    confidence_score: float = 1.0
    processing_flags: List[str] = field(default_factory=list)

    def get_effective_date(self) -> Optional[datetime]:
        """Get the effective date for this chunk"""
        if self.source_date:
            return self.source_date
        if self.temporal_signals and self.temporal_signals.inferred_date:
            return self.temporal_signals.inferred_date
        return self.extraction_date


# =============================================================================
# Gold Layer Models - Entity Versioning
# =============================================================================

@dataclass
class VersionedAttribute:
    """
    A single versioned attribute value.
    Tracks when and where an attribute value was recorded.
    """

    attribute_name: str                # "budget", "status", "manager"
    value: Any                         # The attribute value
    timestamp: datetime                # When this value was recorded
    source_doc_id: str                 # Which document stated this
    source_chunk_id: Optional[str] = None
    confidence: float = 1.0            # Confidence in this value

    # Context for the value
    evidence_text: Optional[str] = None  # Text that supports this value

    def __lt__(self, other: "VersionedAttribute") -> bool:
        """Compare by timestamp for sorting"""
        return self.timestamp < other.timestamp


@dataclass
class AttributeConflict:
    """
    Detected contradiction between sources for an entity attribute.
    """

    conflict_id: str                   # Unique identifier
    entity_id: str                     # Entity with conflict
    entity_name: str                   # Human-readable name
    attribute_name: str                # Which attribute conflicts

    # Conflicting values
    value_a: Any
    value_b: Any

    # Source information
    source_a_doc_id: str
    source_b_doc_id: str
    timestamp_a: datetime
    timestamp_b: datetime
    confidence_a: float
    confidence_b: float

    # Evidence
    evidence_a: Optional[str] = None   # Supporting text for value_a
    evidence_b: Optional[str] = None   # Supporting text for value_b

    # Resolution
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.NEWER_WINS
    resolved_value: Optional[Any] = None
    resolution_confidence: float = 0.0
    resolution_explanation: Optional[str] = None
    is_resolved: bool = False
    resolved_by: Optional[str] = None  # "system" or "user"

    def to_summary(self) -> str:
        """Generate human-readable conflict summary"""
        return (
            f"Conflict in '{self.attribute_name}' for {self.entity_name}: "
            f"'{self.value_a}' ({self.timestamp_a.strftime('%Y-%m-%d')}) vs "
            f"'{self.value_b}' ({self.timestamp_b.strftime('%Y-%m-%d')})"
        )


@dataclass
class VersionedEntity:
    """
    Entity with complete version history for all attributes.
    Enables tracking changes over time and detecting conflicts.
    """

    entity_id: str
    entity_type: str                   # PERSON, PROJECT, ORGANIZATION, etc.
    name: str                          # Primary name
    aliases: List[str] = field(default_factory=list)

    # Current attributes (most recent values)
    current_attributes: Dict[str, Any] = field(default_factory=dict)

    # Version history for each attribute
    # Key: attribute_name, Value: list of VersionedAttribute sorted by time
    attribute_history: Dict[str, List[VersionedAttribute]] = field(default_factory=dict)

    # Detected conflicts
    conflicts: List[AttributeConflict] = field(default_factory=list)

    # Metadata
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source_doc_ids: List[str] = field(default_factory=list)

    def add_attribute(self, attr: VersionedAttribute) -> None:
        """Add a new attribute value to history"""
        if attr.attribute_name not in self.attribute_history:
            self.attribute_history[attr.attribute_name] = []

        self.attribute_history[attr.attribute_name].append(attr)
        self.attribute_history[attr.attribute_name].sort()

        # Update current value if this is the newest
        history = self.attribute_history[attr.attribute_name]
        if history and history[-1] == attr:
            self.current_attributes[attr.attribute_name] = attr.value

        # Update timestamps
        if self.first_seen is None or attr.timestamp < self.first_seen:
            self.first_seen = attr.timestamp
        if self.last_updated is None or attr.timestamp > self.last_updated:
            self.last_updated = attr.timestamp

    def get_attribute_at_time(
        self,
        attribute_name: str,
        at_time: datetime
    ) -> Optional[Any]:
        """Get attribute value as of a specific time"""
        if attribute_name not in self.attribute_history:
            return None

        # Find the most recent value before at_time
        history = self.attribute_history[attribute_name]
        valid_values = [v for v in history if v.timestamp <= at_time]

        if valid_values:
            return valid_values[-1].value
        return None

    def has_conflicts(self) -> bool:
        """Check if entity has unresolved conflicts"""
        return any(not c.is_resolved for c in self.conflicts)


# =============================================================================
# Retrieval Layer Models
# =============================================================================

@dataclass
class RetrievalResult:
    """Single retrieval result with temporal metadata"""

    chunk_id: str
    doc_id: str
    text: str

    # Scores
    base_score: float                  # Original retrieval score
    temporal_weight: float = 1.0       # Temporal decay weight
    final_score: float = 0.0           # base_score * temporal_weight

    # Temporal info
    doc_date: Optional[datetime] = None

    # Source info
    source_file: Optional[str] = None

    def __post_init__(self):
        self.final_score = self.base_score * self.temporal_weight


@dataclass
class ConflictAwareContext:
    """
    Retrieval context that includes conflict information.
    Returned by ConflictAwareRetriever.
    """

    # Retrieved content
    chunks: List[RetrievalResult] = field(default_factory=list)
    entities: List[VersionedEntity] = field(default_factory=list)

    # For PathRAG
    paths: List[Any] = field(default_factory=list)  # ReasoningPath objects

    # Conflict information
    conflicts: List[AttributeConflict] = field(default_factory=list)
    conflict_summary: str = ""

    # Metadata
    query: str = ""
    retrieval_timestamp: datetime = field(default_factory=datetime.now)

    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected"""
        return len(self.conflicts) > 0

    def get_unresolved_conflicts(self) -> List[AttributeConflict]:
        """Get conflicts that need user resolution"""
        return [c for c in self.conflicts if not c.is_resolved]

    def to_prompt_context(self) -> str:
        """Format context for LLM prompt"""
        context = "## Retrieved Information\n\n"

        # Add chunks
        for i, chunk in enumerate(self.chunks[:10], 1):
            context += f"### Source {i}"
            if chunk.doc_date:
                context += f" ({chunk.doc_date.strftime('%Y-%m-%d')})"
            context += f"\n{chunk.text}\n\n"

        # Add conflict warning if present
        if self.conflict_summary:
            context += f"\n{self.conflict_summary}\n"

        return context


# =============================================================================
# Evaluation Models
# =============================================================================

@dataclass
class PIIEvaluationResult:
    """Results of PII anonymization evaluation"""

    # Counts
    total_ground_truth: int
    total_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int

    # Metrics
    precision: float
    recall: float
    f1_score: float

    # By type
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Detailed results
    tp_details: List[PIIAnnotation] = field(default_factory=list)
    fp_details: List[PIIAnnotation] = field(default_factory=list)
    fn_details: List[PIIAnnotation] = field(default_factory=list)

    def meets_threshold(
        self,
        recall_threshold: float = 0.95,
        precision_threshold: float = 0.90
    ) -> bool:
        """Check if results meet quality thresholds"""
        return self.recall >= recall_threshold and self.precision >= precision_threshold

    def to_report(self) -> str:
        """Generate evaluation report"""
        status_recall = "✓" if self.recall >= 0.95 else "✗"
        status_precision = "✓" if self.precision >= 0.90 else "✗"
        status_f1 = "✓" if self.f1_score >= 0.92 else "✗"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              PII ANONYMIZATION EVALUATION REPORT                 ║
╠══════════════════════════════════════════════════════════════════╣

SUMMARY
───────
Total PII in Ground Truth:  {self.total_ground_truth}
Total PII Detected:         {self.total_detected}

METRICS
───────
┌─────────────┬─────────┬────────┬────────┐
│ Metric      │ Value   │ Target │ Status │
├─────────────┼─────────┼────────┼────────┤
│ Recall      │ {self.recall:.2%}  │ >95%   │ {status_recall}      │
│ Precision   │ {self.precision:.2%}  │ >90%   │ {status_precision}      │
│ F1 Score    │ {self.f1_score:.2%}  │ >92%   │ {status_f1}      │
└─────────────┴─────────┴────────┴────────┘

CONFUSION MATRIX
────────────────
True Positives (correctly detected):   {self.true_positives}
False Positives (incorrectly flagged): {self.false_positives}
False Negatives (missed PII):          {self.false_negatives}

"""

        if self.metrics_by_type:
            report += "BY PII TYPE\n───────────\n"
            for pii_type, metrics in self.metrics_by_type.items():
                report += f"  {pii_type}: P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, F1={metrics['f1']:.2%}\n"

        return report
