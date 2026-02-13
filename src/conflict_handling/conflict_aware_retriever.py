"""
Conflict-Aware Retriever

A retrieval wrapper that applies temporal decay and includes
conflict information in the retrieval context.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol

from .models import (
    ConflictAwareContext,
    RetrievalResult,
    VersionedEntity,
    AttributeConflict,
)
from .conflict_detector import ConflictDetector
from .temporal_decay import TemporalDecayScorer, DecayConfig, AdaptiveDecayScorer
from .entity_versioning import EntityVersionManager


class BaseRetriever(Protocol):
    """Protocol for base retrievers (GraphRAG, PathRAG, etc.)"""

    def retrieve(self, query: str, top_k: int = 10) -> Any:
        """Retrieve relevant context for a query"""
        ...


class ConflictAwareRetriever:
    """
    Retriever wrapper that adds conflict awareness and temporal decay.

    Wraps any base retriever (GraphRAG, PathRAG, Vector) and:
    1. Applies temporal decay to retrieved chunks
    2. Detects conflicts in retrieved entities
    3. Formats conflict information for the LLM
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        entity_manager: Optional[EntityVersionManager] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        temporal_scorer: Optional[TemporalDecayScorer] = None,
        use_adaptive_decay: bool = True
    ):
        """
        Initialize the conflict-aware retriever.

        Args:
            base_retriever: The underlying retriever (GraphRAG, PathRAG, etc.)
            entity_manager: Entity version manager with history
            conflict_detector: Conflict detection component
            temporal_scorer: Temporal decay scorer
            use_adaptive_decay: Use query-adaptive decay
        """
        self.base_retriever = base_retriever
        self.entity_manager = entity_manager or EntityVersionManager()
        self.conflict_detector = conflict_detector or ConflictDetector()

        if use_adaptive_decay:
            self.temporal_scorer = AdaptiveDecayScorer()
        else:
            self.temporal_scorer = temporal_scorer or TemporalDecayScorer()

        self.use_adaptive_decay = use_adaptive_decay

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_conflicts: bool = True,
        apply_temporal_decay: bool = True
    ) -> ConflictAwareContext:
        """
        Retrieve context with conflict awareness.

        Args:
            query: The search query
            top_k: Number of results to return
            include_conflicts: Whether to detect and include conflicts
            apply_temporal_decay: Whether to apply temporal decay

        Returns:
            ConflictAwareContext with chunks, entities, and conflicts
        """
        # Step 1: Base retrieval
        base_result = self.base_retriever.retrieve(query, top_k=top_k * 2)

        # Step 2: Convert to RetrievalResults
        chunks = self._convert_to_results(base_result)

        # Step 3: Apply temporal decay
        if apply_temporal_decay:
            if self.use_adaptive_decay and isinstance(self.temporal_scorer, AdaptiveDecayScorer):
                chunks = self.temporal_scorer.apply_adaptive_decay(chunks, query)
            else:
                chunks = self.temporal_scorer.apply_decay_to_results(chunks)

        # Take top_k after decay adjustment
        chunks = chunks[:top_k]

        # Step 4: Get entities from results
        entities = self._get_entities_from_results(base_result)

        # Step 5: Detect conflicts
        conflicts = []
        conflict_summary = ""

        if include_conflicts and entities:
            conflicts = self._detect_conflicts_in_results(entities)
            conflict_summary = self._generate_conflict_summary(conflicts)

        # Step 6: Get paths if available (PathRAG)
        paths = self._get_paths_from_results(base_result)

        # Step 7: Build context
        context = ConflictAwareContext(
            chunks=chunks,
            entities=entities,
            paths=paths,
            conflicts=conflicts,
            conflict_summary=conflict_summary,
            query=query,
            retrieval_timestamp=datetime.now()
        )

        return context

    def retrieve_with_history(
        self,
        query: str,
        entity_name: str,
        attribute: Optional[str] = None,
        top_k: int = 10
    ) -> ConflictAwareContext:
        """
        Retrieve with full attribute history for an entity.

        Useful for queries like "What was the budget for Project X?"
        where history matters.

        Args:
            query: The search query
            entity_name: Entity to get history for
            attribute: Specific attribute (all if None)
            top_k: Number of results

        Returns:
            Context with entity history
        """
        # Base retrieval
        context = self.retrieve(query, top_k=top_k)

        # Find the entity
        entity = self._find_entity_by_name(entity_name)

        if entity:
            # Get timeline
            timeline = self.entity_manager.get_entity_timeline(
                entity.entity_id,
                attribute_name=attribute
            )

            # Add timeline to context metadata
            context.entity_timeline = timeline

        return context

    def _convert_to_results(self, base_result: Any) -> List[RetrievalResult]:
        """Convert base retriever results to RetrievalResults"""
        results = []

        # Handle different result formats
        if hasattr(base_result, 'chunks'):
            # GraphRAG/PathRAG style
            for chunk in base_result.chunks:
                result = RetrievalResult(
                    chunk_id=getattr(chunk, 'chunk_id', str(id(chunk))),
                    doc_id=getattr(chunk, 'doc_id', ''),
                    text=getattr(chunk, 'text', str(chunk)),
                    base_score=getattr(chunk, 'score', 1.0),
                    doc_date=getattr(chunk, 'source_date', None),
                    source_file=getattr(chunk, 'source_file', None)
                )
                results.append(result)

        elif isinstance(base_result, list):
            # List of chunks/documents
            for i, item in enumerate(base_result):
                if isinstance(item, dict):
                    result = RetrievalResult(
                        chunk_id=item.get('chunk_id', str(i)),
                        doc_id=item.get('doc_id', ''),
                        text=item.get('text', ''),
                        base_score=item.get('score', 1.0),
                        doc_date=item.get('date'),
                        source_file=item.get('source')
                    )
                else:
                    result = RetrievalResult(
                        chunk_id=str(i),
                        doc_id='',
                        text=str(item),
                        base_score=1.0
                    )
                results.append(result)

        return results

    def _get_entities_from_results(
        self,
        base_result: Any
    ) -> List[VersionedEntity]:
        """Extract entities from retrieval results"""
        entities = []

        # Try to get entities from result
        if hasattr(base_result, 'entities'):
            for entity in base_result.entities:
                # Get versioned entity from manager
                if hasattr(entity, 'entity_id'):
                    versioned = self.entity_manager.entities.get(entity.entity_id)
                    if versioned:
                        entities.append(versioned)
                elif hasattr(entity, 'name'):
                    # Look up by name
                    found = self._find_entity_by_name(entity.name)
                    if found:
                        entities.append(found)

        return entities

    def _get_paths_from_results(self, base_result: Any) -> List[Any]:
        """Extract paths from PathRAG results"""
        if hasattr(base_result, 'paths'):
            return base_result.paths
        return []

    def _detect_conflicts_in_results(
        self,
        entities: List[VersionedEntity]
    ) -> List[AttributeConflict]:
        """Detect conflicts in retrieved entities"""
        all_conflicts = []

        for entity in entities:
            conflicts = self.conflict_detector.detect_conflicts(entity)
            all_conflicts.extend(conflicts)

        return all_conflicts

    def _generate_conflict_summary(
        self,
        conflicts: List[AttributeConflict]
    ) -> str:
        """Generate human-readable conflict summary"""
        if not conflicts:
            return ""

        unresolved = [c for c in conflicts if not c.is_resolved]

        if not unresolved:
            # All conflicts resolved
            return ""

        summary = "**Note: The following information has conflicting values:**\n\n"

        for c in unresolved[:5]:  # Limit to 5 conflicts
            summary += f"- **{c.entity_name}** ({c.attribute_name}): "
            summary += f"'{c.value_a}' ({c.timestamp_a.strftime('%Y-%m')}) vs "
            summary += f"'{c.value_b}' ({c.timestamp_b.strftime('%Y-%m')})\n"

        if len(unresolved) > 5:
            summary += f"\n*...and {len(unresolved) - 5} more conflicts*\n"

        return summary

    def _find_entity_by_name(self, name: str) -> Optional[VersionedEntity]:
        """Find entity by name in the manager"""
        normalized = name.lower().strip()

        for entity in self.entity_manager.entities.values():
            if entity.name.lower() == normalized:
                return entity
            if normalized in [a.lower() for a in entity.aliases]:
                return entity

        return None


class ConflictAwareRetrieverBuilder:
    """Builder for creating configured ConflictAwareRetriever instances"""

    def __init__(self):
        self._base_retriever = None
        self._entity_manager = None
        self._conflict_detector = None
        self._temporal_scorer = None
        self._use_adaptive_decay = True

    def with_base_retriever(self, retriever: BaseRetriever) -> "ConflictAwareRetrieverBuilder":
        """Set the base retriever"""
        self._base_retriever = retriever
        return self

    def with_entity_manager(self, manager: EntityVersionManager) -> "ConflictAwareRetrieverBuilder":
        """Set the entity manager"""
        self._entity_manager = manager
        return self

    def with_conflict_detector(self, detector: ConflictDetector) -> "ConflictAwareRetrieverBuilder":
        """Set the conflict detector"""
        self._conflict_detector = detector
        return self

    def with_temporal_decay(
        self,
        half_life_days: int = 365,
        min_weight: float = 0.1,
        decay_type: str = "exponential"
    ) -> "ConflictAwareRetrieverBuilder":
        """Configure temporal decay"""
        config = DecayConfig(
            half_life_days=half_life_days,
            min_weight=min_weight,
            decay_type=decay_type
        )
        self._temporal_scorer = TemporalDecayScorer(config)
        self._use_adaptive_decay = False
        return self

    def with_adaptive_decay(self) -> "ConflictAwareRetrieverBuilder":
        """Use adaptive temporal decay"""
        self._use_adaptive_decay = True
        return self

    def build(self) -> ConflictAwareRetriever:
        """Build the configured retriever"""
        if self._base_retriever is None:
            raise ValueError("Base retriever is required")

        return ConflictAwareRetriever(
            base_retriever=self._base_retriever,
            entity_manager=self._entity_manager,
            conflict_detector=self._conflict_detector,
            temporal_scorer=self._temporal_scorer,
            use_adaptive_decay=self._use_adaptive_decay
        )


# Convenience function
def wrap_retriever_with_conflict_awareness(
    base_retriever: BaseRetriever,
    entity_manager: Optional[EntityVersionManager] = None,
    half_life_days: int = 365
) -> ConflictAwareRetriever:
    """
    Wrap a base retriever with conflict awareness.

    Args:
        base_retriever: The retriever to wrap
        entity_manager: Entity manager with version history
        half_life_days: Temporal decay half-life

    Returns:
        ConflictAwareRetriever wrapping the base retriever
    """
    return (
        ConflictAwareRetrieverBuilder()
        .with_base_retriever(base_retriever)
        .with_entity_manager(entity_manager)
        .with_temporal_decay(half_life_days=half_life_days)
        .build()
    )
