"""
Conflict Detector

Detects contradictions in entity attributes across different sources
and timestamps. Identifies conflicts that need resolution.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from .models import (
    VersionedEntity,
    VersionedAttribute,
    AttributeConflict,
    ConflictResolutionStrategy,
)


class ConflictDetector:
    """
    Detect and analyze conflicts in entity attributes.

    Examines version history of entity attributes to identify
    contradictions that may require resolution.
    """

    # Attributes that can have meaningful conflicts
    CONFLICTABLE_ATTRIBUTES = [
        # Financial
        "budget", "cost", "price", "revenue", "salary", "funding",
        # Status
        "status", "state", "phase", "approval_status",
        # Temporal
        "deadline", "due_date", "start_date", "end_date", "eta",
        # Organizational
        "manager", "lead", "owner", "responsible", "team_size", "headcount",
        # Location
        "location", "office", "address", "site",
        # Quantitative
        "quantity", "count", "percentage", "rate", "score",
        # Decisions
        "decision", "outcome", "result", "verdict",
    ]

    # Attributes where changes are expected (not conflicts)
    EXPECTED_CHANGE_ATTRIBUTES = [
        "status", "state", "phase",  # Status naturally progresses
        "last_updated", "modified_date",  # Metadata
    ]

    # Minimum time gap to consider values from different "versions"
    MIN_TIME_GAP_HOURS = 1

    def __init__(
        self,
        conflictable_attributes: Optional[List[str]] = None,
        numeric_threshold: float = 0.2,
        auto_resolve_by_recency: bool = True
    ):
        """
        Initialize the conflict detector.

        Args:
            conflictable_attributes: Custom list of attributes to check
            numeric_threshold: Threshold for numeric conflicts (20% by default)
            auto_resolve_by_recency: Auto-resolve using "newer wins" strategy
        """
        self.conflictable_attributes = (
            conflictable_attributes or self.CONFLICTABLE_ATTRIBUTES
        )
        self.numeric_threshold = numeric_threshold
        self.auto_resolve_by_recency = auto_resolve_by_recency

    def detect_conflicts(
        self,
        entity: VersionedEntity
    ) -> List[AttributeConflict]:
        """
        Detect conflicts in an entity's attribute history.

        Args:
            entity: Entity with version history to analyze

        Returns:
            List of detected conflicts
        """
        conflicts = []

        for attr_name in self.conflictable_attributes:
            if attr_name not in entity.attribute_history:
                continue

            history = entity.attribute_history[attr_name]

            # Need at least 2 values to have a conflict
            if len(history) < 2:
                continue

            # Sort by timestamp
            sorted_history = sorted(history, key=lambda x: x.timestamp)

            # Check consecutive values for contradictions
            attr_conflicts = self._check_attribute_conflicts(
                entity, attr_name, sorted_history
            )
            conflicts.extend(attr_conflicts)

        # Store conflicts on entity
        entity.conflicts = conflicts

        return conflicts

    def detect_conflicts_batch(
        self,
        entities: List[VersionedEntity]
    ) -> Dict[str, List[AttributeConflict]]:
        """
        Detect conflicts across multiple entities.

        Args:
            entities: List of entities to analyze

        Returns:
            Dictionary mapping entity_id to list of conflicts
        """
        all_conflicts = {}

        for entity in entities:
            conflicts = self.detect_conflicts(entity)
            if conflicts:
                all_conflicts[entity.entity_id] = conflicts

        return all_conflicts

    def _check_attribute_conflicts(
        self,
        entity: VersionedEntity,
        attr_name: str,
        sorted_history: List[VersionedAttribute]
    ) -> List[AttributeConflict]:
        """Check for conflicts in a single attribute's history"""
        conflicts = []

        for i in range(len(sorted_history) - 1):
            older = sorted_history[i]
            newer = sorted_history[i + 1]

            # Check if there's enough time gap
            time_gap = newer.timestamp - older.timestamp
            if time_gap < timedelta(hours=self.MIN_TIME_GAP_HOURS):
                continue  # Same update session, not a conflict

            # Check if values contradict
            is_conflict, conflict_type = self._is_contradiction(
                older.value, newer.value, attr_name
            )

            if is_conflict:
                conflict = self._create_conflict(
                    entity, attr_name, older, newer, conflict_type
                )
                conflicts.append(conflict)

        return conflicts

    def _is_contradiction(
        self,
        value_a: Any,
        value_b: Any,
        attr_name: str
    ) -> Tuple[bool, str]:
        """
        Determine if two values contradict each other.

        Returns:
            Tuple of (is_conflict, conflict_type)
        """
        # Same value = no conflict
        if value_a == value_b:
            return False, ""

        # None values = not enough info to conflict
        if value_a is None or value_b is None:
            return False, ""

        # Expected changes are not conflicts
        if attr_name in self.EXPECTED_CHANGE_ATTRIBUTES:
            return False, ""

        # Numeric attributes: check for significant change
        if self._is_numeric_attribute(attr_name):
            try:
                a = self._extract_numeric(value_a)
                b = self._extract_numeric(value_b)

                if a is not None and b is not None and max(abs(a), abs(b)) > 0:
                    change_ratio = abs(a - b) / max(abs(a), abs(b))
                    if change_ratio > self.numeric_threshold:
                        return True, "numeric_change"
                    else:
                        return False, ""  # Small change, not a conflict
            except (ValueError, TypeError):
                pass

        # String comparison for other attributes
        str_a = str(value_a).lower().strip()
        str_b = str(value_b).lower().strip()

        if str_a != str_b:
            return True, "value_change"

        return False, ""

    def _is_numeric_attribute(self, attr_name: str) -> bool:
        """Check if attribute typically holds numeric values"""
        numeric_keywords = [
            "budget", "cost", "price", "revenue", "salary", "funding",
            "quantity", "count", "percentage", "rate", "score", "size",
            "headcount", "team_size"
        ]
        return any(kw in attr_name.lower() for kw in numeric_keywords)

    def _extract_numeric(self, value: Any) -> Optional[float]:
        """Extract numeric value from various formats"""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove currency symbols and common suffixes
            import re
            cleaned = re.sub(r"[$€£¥,\s]", "", value)
            cleaned = re.sub(r"[kKmMbB]$", lambda m: {
                'k': '000', 'K': '000',
                'm': '000000', 'M': '000000',
                'b': '000000000', 'B': '000000000'
            }.get(m.group(), ''), cleaned)

            try:
                return float(cleaned)
            except ValueError:
                return None

        return None

    def _create_conflict(
        self,
        entity: VersionedEntity,
        attr_name: str,
        older: VersionedAttribute,
        newer: VersionedAttribute,
        conflict_type: str
    ) -> AttributeConflict:
        """Create a conflict record"""
        conflict = AttributeConflict(
            conflict_id=str(uuid.uuid4())[:8],
            entity_id=entity.entity_id,
            entity_name=entity.name,
            attribute_name=attr_name,
            value_a=older.value,
            value_b=newer.value,
            source_a_doc_id=older.source_doc_id,
            source_b_doc_id=newer.source_doc_id,
            timestamp_a=older.timestamp,
            timestamp_b=newer.timestamp,
            confidence_a=older.confidence,
            confidence_b=newer.confidence,
            evidence_a=older.evidence_text,
            evidence_b=newer.evidence_text,
        )

        # Auto-resolve if enabled
        if self.auto_resolve_by_recency:
            self._auto_resolve(conflict, older, newer)

        return conflict

    def _auto_resolve(
        self,
        conflict: AttributeConflict,
        older: VersionedAttribute,
        newer: VersionedAttribute
    ) -> None:
        """Auto-resolve conflict using newer wins strategy"""
        conflict.resolution_strategy = ConflictResolutionStrategy.NEWER_WINS
        conflict.resolved_value = newer.value
        conflict.resolution_confidence = self._calc_resolution_confidence(
            older, newer
        )
        conflict.resolution_explanation = (
            f"Automatically resolved using newer value from "
            f"{newer.timestamp.strftime('%Y-%m-%d')}. "
            f"Previous value '{older.value}' from "
            f"{older.timestamp.strftime('%Y-%m-%d')} superseded."
        )
        conflict.is_resolved = True
        conflict.resolved_by = "system"

    def _calc_resolution_confidence(
        self,
        older: VersionedAttribute,
        newer: VersionedAttribute
    ) -> float:
        """Calculate confidence in resolution"""
        confidence = 0.5  # Base confidence

        # Time gap increases confidence
        time_gap = (newer.timestamp - older.timestamp).days
        if time_gap > 30:
            confidence += 0.2
        elif time_gap > 7:
            confidence += 0.1

        # Higher newer confidence increases resolution confidence
        if newer.confidence > older.confidence:
            confidence += 0.15
        elif newer.confidence == older.confidence:
            confidence += 0.05

        # Evidence text increases confidence
        if newer.evidence_text:
            confidence += 0.1

        return min(confidence, 1.0)

    def resolve_conflict(
        self,
        conflict: AttributeConflict,
        strategy: ConflictResolutionStrategy,
        manual_value: Optional[Any] = None,
        explanation: Optional[str] = None
    ) -> AttributeConflict:
        """
        Manually resolve a conflict.

        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy to use
            manual_value: Value to use for USER_RESOLUTION strategy
            explanation: Explanation for the resolution

        Returns:
            Updated conflict with resolution
        """
        conflict.resolution_strategy = strategy

        if strategy == ConflictResolutionStrategy.NEWER_WINS:
            conflict.resolved_value = conflict.value_b
        elif strategy == ConflictResolutionStrategy.OLDER_WINS:
            conflict.resolved_value = conflict.value_a
        elif strategy == ConflictResolutionStrategy.HIGHER_CONFIDENCE:
            if conflict.confidence_b >= conflict.confidence_a:
                conflict.resolved_value = conflict.value_b
            else:
                conflict.resolved_value = conflict.value_a
        elif strategy == ConflictResolutionStrategy.AVERAGE:
            # Only for numeric values
            try:
                a = self._extract_numeric(conflict.value_a)
                b = self._extract_numeric(conflict.value_b)
                if a is not None and b is not None:
                    conflict.resolved_value = (a + b) / 2
                else:
                    conflict.resolved_value = conflict.value_b
            except (ValueError, TypeError):
                conflict.resolved_value = conflict.value_b
        elif strategy == ConflictResolutionStrategy.USER_RESOLUTION:
            if manual_value is not None:
                conflict.resolved_value = manual_value
            else:
                raise ValueError("manual_value required for USER_RESOLUTION strategy")
        elif strategy == ConflictResolutionStrategy.BOTH_VALID:
            # Both values are kept valid (context-dependent)
            conflict.resolved_value = {
                "value_a": conflict.value_a,
                "timestamp_a": conflict.timestamp_a,
                "value_b": conflict.value_b,
                "timestamp_b": conflict.timestamp_b,
            }

        conflict.resolution_explanation = explanation or (
            f"Resolved using {strategy.value} strategy."
        )
        conflict.is_resolved = True
        conflict.resolved_by = "user"

        return conflict

    def get_conflict_summary(
        self,
        conflicts: List[AttributeConflict]
    ) -> str:
        """
        Generate human-readable summary of conflicts.

        Args:
            conflicts: List of conflicts to summarize

        Returns:
            Formatted string summary
        """
        if not conflicts:
            return "No conflicts detected."

        summary = f"**{len(conflicts)} conflict(s) detected:**\n\n"

        for i, c in enumerate(conflicts, 1):
            status = "✓ Resolved" if c.is_resolved else "⚠️ Unresolved"
            summary += f"{i}. **{c.entity_name}** - {c.attribute_name} {status}\n"
            summary += f"   - '{c.value_a}' ({c.timestamp_a.strftime('%Y-%m-%d')})\n"
            summary += f"   - '{c.value_b}' ({c.timestamp_b.strftime('%Y-%m-%d')})\n"

            if c.is_resolved:
                summary += f"   → Resolved: '{c.resolved_value}'\n"

            summary += "\n"

        return summary


# Convenience function
def detect_entity_conflicts(entity: VersionedEntity) -> List[AttributeConflict]:
    """
    Detect conflicts in an entity's attribute history.

    Args:
        entity: Entity to analyze

    Returns:
        List of detected conflicts
    """
    detector = ConflictDetector()
    return detector.detect_conflicts(entity)
