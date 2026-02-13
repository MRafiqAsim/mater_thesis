# Conflict Handling Module
# Handles conflicting information in documents through versioning,
# temporal decay, and conflict detection/resolution

from .models import (
    DocumentMetadata,
    TemporalSignals,
    VersionedAttribute,
    VersionedEntity,
    AttributeConflict,
    ConflictAwareContext,
    EnrichedChunk,
)

from .temporal_extractor import TemporalExpressionExtractor
from .metadata_extractor import MetadataExtractor
from .conflict_detector import ConflictDetector
from .temporal_decay import TemporalDecayScorer
from .conflict_aware_retriever import ConflictAwareRetriever
from .entity_versioning import EntityVersionManager

__all__ = [
    # Models
    "DocumentMetadata",
    "TemporalSignals",
    "VersionedAttribute",
    "VersionedEntity",
    "AttributeConflict",
    "ConflictAwareContext",
    "EnrichedChunk",
    # Extractors
    "TemporalExpressionExtractor",
    "MetadataExtractor",
    # Core components
    "ConflictDetector",
    "TemporalDecayScorer",
    "ConflictAwareRetriever",
    "EntityVersionManager",
]
