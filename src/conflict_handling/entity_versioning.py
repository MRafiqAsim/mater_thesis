"""
Entity Version Manager

Manages versioned entities with attribute history tracking.
Supports building entities from multiple sources and tracking
changes over time.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .models import (
    VersionedEntity,
    VersionedAttribute,
    AttributeConflict,
    EnrichedChunk,
    EntityMention,
)
from .conflict_detector import ConflictDetector


class EntityVersionManager:
    """
    Manages the creation and updating of versioned entities.

    Tracks attribute changes across multiple documents and
    maintains a complete version history for each entity.
    """

    def __init__(
        self,
        conflict_detector: Optional[ConflictDetector] = None,
        merge_threshold: float = 0.85
    ):
        """
        Initialize the entity version manager.

        Args:
            conflict_detector: Detector for finding conflicts
            merge_threshold: Similarity threshold for entity merging
        """
        self.entities: Dict[str, VersionedEntity] = {}
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.merge_threshold = merge_threshold

        # Entity name to ID mapping for resolution
        self._name_to_id: Dict[str, str] = {}
        self._alias_to_id: Dict[str, str] = {}

    def get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        source_doc_id: str
    ) -> VersionedEntity:
        """
        Get existing entity or create new one.

        Args:
            name: Entity name
            entity_type: Type (PERSON, PROJECT, etc.)
            source_doc_id: Source document ID

        Returns:
            Existing or new VersionedEntity
        """
        # Normalize name for matching
        normalized = self._normalize_name(name)

        # Check if entity exists
        entity_id = self._find_entity_id(normalized, entity_type)

        if entity_id and entity_id in self.entities:
            entity = self.entities[entity_id]
            # Add source doc if not already tracked
            if source_doc_id not in entity.source_doc_ids:
                entity.source_doc_ids.append(source_doc_id)
            return entity

        # Create new entity
        entity_id = str(uuid.uuid4())[:12]
        entity = VersionedEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            aliases=[normalized] if normalized != name else [],
            source_doc_ids=[source_doc_id],
            first_seen=datetime.now()
        )

        # Register entity
        self.entities[entity_id] = entity
        self._name_to_id[normalized] = entity_id
        self._name_to_id[name.lower()] = entity_id

        return entity

    def add_attribute(
        self,
        entity: VersionedEntity,
        attribute_name: str,
        value: Any,
        timestamp: datetime,
        source_doc_id: str,
        source_chunk_id: Optional[str] = None,
        confidence: float = 1.0,
        evidence_text: Optional[str] = None
    ) -> VersionedAttribute:
        """
        Add an attribute value to an entity's history.

        Args:
            entity: The entity to update
            attribute_name: Name of the attribute
            value: Attribute value
            timestamp: When this value was recorded
            source_doc_id: Source document
            source_chunk_id: Source chunk (optional)
            confidence: Confidence in this value
            evidence_text: Text evidence supporting this value

        Returns:
            The created VersionedAttribute
        """
        attr = VersionedAttribute(
            attribute_name=attribute_name,
            value=value,
            timestamp=timestamp,
            source_doc_id=source_doc_id,
            source_chunk_id=source_chunk_id,
            confidence=confidence,
            evidence_text=evidence_text
        )

        entity.add_attribute(attr)

        return attr

    def add_attributes_from_extraction(
        self,
        entity: VersionedEntity,
        attributes: Dict[str, Any],
        timestamp: datetime,
        source_doc_id: str,
        source_chunk_id: Optional[str] = None,
        confidence: float = 1.0
    ) -> List[VersionedAttribute]:
        """
        Add multiple attributes from an extraction result.

        Args:
            entity: The entity to update
            attributes: Dictionary of attribute name -> value
            timestamp: When these were recorded
            source_doc_id: Source document
            source_chunk_id: Source chunk
            confidence: Confidence in these values

        Returns:
            List of created VersionedAttributes
        """
        created = []

        for attr_name, value in attributes.items():
            if value is not None:
                attr = self.add_attribute(
                    entity=entity,
                    attribute_name=attr_name,
                    value=value,
                    timestamp=timestamp,
                    source_doc_id=source_doc_id,
                    source_chunk_id=source_chunk_id,
                    confidence=confidence
                )
                created.append(attr)

        return created

    def process_chunk(
        self,
        chunk: EnrichedChunk,
        extracted_attributes: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[VersionedEntity]:
        """
        Process a chunk and update entity versions.

        Args:
            chunk: Enriched chunk with entities
            extracted_attributes: Optional dict of entity_name -> attributes

        Returns:
            List of updated entities
        """
        updated_entities = []
        timestamp = chunk.get_effective_date() or datetime.now()

        for mention in chunk.entities:
            # Get or create entity
            entity = self.get_or_create_entity(
                name=mention.text,
                entity_type=mention.entity_type,
                source_doc_id=chunk.doc_id
            )

            # Add any extracted attributes
            if extracted_attributes and mention.text in extracted_attributes:
                attrs = extracted_attributes[mention.text]
                self.add_attributes_from_extraction(
                    entity=entity,
                    attributes=attrs,
                    timestamp=timestamp,
                    source_doc_id=chunk.doc_id,
                    source_chunk_id=chunk.chunk_id,
                    confidence=mention.confidence
                )

            updated_entities.append(entity)

        return updated_entities

    def merge_entities(
        self,
        entity_a: VersionedEntity,
        entity_b: VersionedEntity
    ) -> VersionedEntity:
        """
        Merge two entities that refer to the same real-world entity.

        Args:
            entity_a: First entity (will be kept)
            entity_b: Second entity (will be merged into first)

        Returns:
            Merged entity (entity_a with entity_b's data)
        """
        # Add aliases
        entity_a.aliases.append(entity_b.name)
        entity_a.aliases.extend(entity_b.aliases)
        entity_a.aliases = list(set(entity_a.aliases))

        # Merge attribute histories
        for attr_name, history in entity_b.attribute_history.items():
            for attr in history:
                entity_a.add_attribute(attr)

        # Merge source docs
        entity_a.source_doc_ids.extend(entity_b.source_doc_ids)
        entity_a.source_doc_ids = list(set(entity_a.source_doc_ids))

        # Update timestamps
        if entity_b.first_seen and (
            not entity_a.first_seen or entity_b.first_seen < entity_a.first_seen
        ):
            entity_a.first_seen = entity_b.first_seen

        if entity_b.last_updated and (
            not entity_a.last_updated or entity_b.last_updated > entity_a.last_updated
        ):
            entity_a.last_updated = entity_b.last_updated

        # Update name mappings
        for alias in entity_b.aliases + [entity_b.name]:
            normalized = self._normalize_name(alias)
            self._name_to_id[normalized] = entity_a.entity_id
            self._alias_to_id[normalized] = entity_a.entity_id

        # Remove merged entity
        if entity_b.entity_id in self.entities:
            del self.entities[entity_b.entity_id]

        return entity_a

    def find_similar_entities(
        self,
        entity: VersionedEntity,
        threshold: Optional[float] = None
    ) -> List[Tuple[VersionedEntity, float]]:
        """
        Find entities similar to the given one (potential duplicates).

        Args:
            entity: Entity to find matches for
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of (entity, similarity_score) tuples
        """
        threshold = threshold or self.merge_threshold
        similar = []

        for other_id, other in self.entities.items():
            if other_id == entity.entity_id:
                continue

            if other.entity_type != entity.entity_type:
                continue

            # Calculate name similarity
            similarity = self._calculate_similarity(entity, other)

            if similarity >= threshold:
                similar.append((other, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def detect_all_conflicts(self) -> Dict[str, List[AttributeConflict]]:
        """
        Detect conflicts across all entities.

        Returns:
            Dictionary of entity_id -> list of conflicts
        """
        return self.conflict_detector.detect_conflicts_batch(
            list(self.entities.values())
        )

    def get_entity_timeline(
        self,
        entity_id: str,
        attribute_name: Optional[str] = None
    ) -> List[VersionedAttribute]:
        """
        Get the timeline of attribute changes for an entity.

        Args:
            entity_id: Entity to get timeline for
            attribute_name: Specific attribute (all if None)

        Returns:
            Chronologically sorted list of attributes
        """
        if entity_id not in self.entities:
            return []

        entity = self.entities[entity_id]
        timeline = []

        if attribute_name:
            # Single attribute timeline
            if attribute_name in entity.attribute_history:
                timeline = entity.attribute_history[attribute_name]
        else:
            # All attributes timeline
            for attr_list in entity.attribute_history.values():
                timeline.extend(attr_list)

        return sorted(timeline, key=lambda x: x.timestamp)

    def get_entity_state_at_time(
        self,
        entity_id: str,
        at_time: datetime
    ) -> Dict[str, Any]:
        """
        Get entity state as of a specific time.

        Args:
            entity_id: Entity ID
            at_time: Point in time

        Returns:
            Dictionary of attribute values as of that time
        """
        if entity_id not in self.entities:
            return {}

        entity = self.entities[entity_id]
        state = {}

        for attr_name in entity.attribute_history:
            value = entity.get_attribute_at_time(attr_name, at_time)
            if value is not None:
                state[attr_name] = value

        return state

    def export_entities(self) -> List[Dict]:
        """
        Export all entities as dictionaries.

        Returns:
            List of entity dictionaries
        """
        exported = []

        for entity in self.entities.values():
            exported.append({
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "aliases": entity.aliases,
                "current_attributes": entity.current_attributes,
                "first_seen": entity.first_seen.isoformat() if entity.first_seen else None,
                "last_updated": entity.last_updated.isoformat() if entity.last_updated else None,
                "source_count": len(entity.source_doc_ids),
                "conflict_count": len([c for c in entity.conflicts if not c.is_resolved]),
                "attribute_history": {
                    attr: [
                        {
                            "value": v.value,
                            "timestamp": v.timestamp.isoformat(),
                            "source": v.source_doc_id,
                            "confidence": v.confidence
                        }
                        for v in versions
                    ]
                    for attr, versions in entity.attribute_history.items()
                }
            })

        return exported

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching"""
        return name.lower().strip()

    def _find_entity_id(
        self,
        normalized_name: str,
        entity_type: str
    ) -> Optional[str]:
        """Find entity ID by normalized name"""
        # Check direct name match
        if normalized_name in self._name_to_id:
            entity_id = self._name_to_id[normalized_name]
            # Verify type matches
            if entity_id in self.entities:
                if self.entities[entity_id].entity_type == entity_type:
                    return entity_id

        # Check aliases
        if normalized_name in self._alias_to_id:
            entity_id = self._alias_to_id[normalized_name]
            if entity_id in self.entities:
                if self.entities[entity_id].entity_type == entity_type:
                    return entity_id

        return None

    def _calculate_similarity(
        self,
        entity_a: VersionedEntity,
        entity_b: VersionedEntity
    ) -> float:
        """Calculate similarity between two entities"""
        # Simple Jaccard similarity on name tokens
        tokens_a = set(entity_a.name.lower().split())
        tokens_b = set(entity_b.name.lower().split())

        # Add alias tokens
        for alias in entity_a.aliases:
            tokens_a.update(alias.lower().split())
        for alias in entity_b.aliases:
            tokens_b.update(alias.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)


# Convenience function
def create_entity_manager() -> EntityVersionManager:
    """Create a new entity version manager with default settings."""
    return EntityVersionManager()
