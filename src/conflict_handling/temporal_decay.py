"""
Temporal Decay Scorer

Applies temporal decay to retrieval scores, weighting recent
documents more heavily than older ones.
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Callable
from dataclasses import dataclass

from .models import RetrievalResult


@dataclass
class DecayConfig:
    """Configuration for temporal decay"""

    # Half-life in days (score halves after this many days)
    half_life_days: int = 365

    # Minimum weight for very old documents
    min_weight: float = 0.1

    # Maximum weight (for recent documents)
    max_weight: float = 1.0

    # Reference date (usually "now")
    reference_date: Optional[datetime] = None

    # Decay function type
    decay_type: str = "exponential"  # "exponential", "linear", "step"

    # For step decay: cutoff dates and weights
    step_cutoffs: Optional[List[tuple]] = None  # [(days, weight), ...]


class TemporalDecayScorer:
    """
    Apply temporal decay to document retrieval scores.

    Recent documents are weighted more heavily than older ones,
    reflecting the assumption that newer information is more
    likely to be current and accurate.
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        """
        Initialize the temporal decay scorer.

        Args:
            config: Decay configuration. Uses defaults if None.
        """
        self.config = config or DecayConfig()
        self.reference_date = (
            self.config.reference_date or datetime.now()
        )

        # Pre-calculate decay constant for exponential decay
        # Formula: weight = 2^(-age/half_life)
        self._decay_constant = math.log(2) / self.config.half_life_days

    def calculate_decay_weight(
        self,
        doc_date: Optional[datetime]
    ) -> float:
        """
        Calculate temporal decay weight for a document.

        Args:
            doc_date: Document date (None = unknown)

        Returns:
            Decay weight between min_weight and max_weight
        """
        # Unknown date = medium weight
        if doc_date is None:
            return (self.config.min_weight + self.config.max_weight) / 2

        # Calculate age in days
        age_days = (self.reference_date - doc_date).days

        # Future date = max weight
        if age_days < 0:
            return self.config.max_weight

        # Apply decay based on type
        if self.config.decay_type == "exponential":
            weight = self._exponential_decay(age_days)
        elif self.config.decay_type == "linear":
            weight = self._linear_decay(age_days)
        elif self.config.decay_type == "step":
            weight = self._step_decay(age_days)
        else:
            weight = self._exponential_decay(age_days)

        # Clamp to configured range
        return max(self.config.min_weight, min(self.config.max_weight, weight))

    def _exponential_decay(self, age_days: int) -> float:
        """
        Exponential decay: weight = 2^(-age/half_life)

        Smooth decay that halves at each half_life interval.
        """
        return math.pow(2, -age_days / self.config.half_life_days)

    def _linear_decay(self, age_days: int) -> float:
        """
        Linear decay: weight decreases linearly to min_weight
        at 2x half_life.
        """
        max_age = self.config.half_life_days * 2
        if age_days >= max_age:
            return self.config.min_weight

        slope = (self.config.max_weight - self.config.min_weight) / max_age
        return self.config.max_weight - (slope * age_days)

    def _step_decay(self, age_days: int) -> float:
        """
        Step decay: discrete weight levels based on age brackets.
        """
        # Default step cutoffs if not configured
        cutoffs = self.config.step_cutoffs or [
            (30, 1.0),     # 0-30 days: full weight
            (90, 0.8),     # 31-90 days: 80%
            (180, 0.6),    # 91-180 days: 60%
            (365, 0.4),    # 181-365 days: 40%
            (730, 0.2),    # 1-2 years: 20%
        ]

        for days, weight in cutoffs:
            if age_days <= days:
                return weight

        return self.config.min_weight

    def apply_decay_to_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Apply temporal decay to a list of retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Results with adjusted scores, re-sorted by final_score
        """
        for result in results:
            # Calculate decay weight
            result.temporal_weight = self.calculate_decay_weight(result.doc_date)

            # Apply to score
            result.final_score = result.base_score * result.temporal_weight

        # Re-sort by adjusted score
        return sorted(results, key=lambda x: x.final_score, reverse=True)

    def apply_decay_to_scores(
        self,
        scores: List[float],
        dates: List[Optional[datetime]]
    ) -> List[float]:
        """
        Apply temporal decay to a list of scores.

        Args:
            scores: Original scores
            dates: Corresponding document dates

        Returns:
            Adjusted scores
        """
        adjusted = []
        for score, date in zip(scores, dates):
            weight = self.calculate_decay_weight(date)
            adjusted.append(score * weight)
        return adjusted

    def get_weight_at_age(self, age_days: int) -> float:
        """
        Get the decay weight for a specific age.

        Useful for debugging and visualization.

        Args:
            age_days: Age in days

        Returns:
            Decay weight
        """
        if age_days < 0:
            return self.config.max_weight
        return self.calculate_decay_weight(
            self.reference_date - timedelta(days=age_days)
        )

    def get_decay_curve(
        self,
        max_age_days: int = 730,
        step_days: int = 30
    ) -> List[tuple]:
        """
        Get decay curve data for visualization.

        Args:
            max_age_days: Maximum age to calculate
            step_days: Step size

        Returns:
            List of (age_days, weight) tuples
        """
        curve = []
        for age in range(0, max_age_days + 1, step_days):
            weight = self.get_weight_at_age(age)
            curve.append((age, weight))
        return curve


class AdaptiveDecayScorer(TemporalDecayScorer):
    """
    Adaptive temporal decay that adjusts based on query context.

    For time-sensitive queries (e.g., "current status"), decay is stronger.
    For historical queries (e.g., "original budget"), decay is weaker.
    """

    # Keywords indicating time-sensitive queries
    TIME_SENSITIVE_KEYWORDS = [
        "current", "latest", "recent", "now", "today",
        "updated", "new", "active", "ongoing"
    ]

    # Keywords indicating historical queries
    HISTORICAL_KEYWORDS = [
        "original", "initial", "first", "historical",
        "previous", "old", "past", "was", "were", "began"
    ]

    def __init__(
        self,
        config: Optional[DecayConfig] = None,
        sensitive_half_life: int = 90,     # 3 months for time-sensitive
        historical_half_life: int = 1825   # 5 years for historical
    ):
        """
        Initialize adaptive decay scorer.

        Args:
            config: Base decay configuration
            sensitive_half_life: Half-life for time-sensitive queries
            historical_half_life: Half-life for historical queries
        """
        super().__init__(config)
        self.sensitive_half_life = sensitive_half_life
        self.historical_half_life = historical_half_life

    def classify_query(self, query: str) -> str:
        """
        Classify query as time-sensitive, historical, or neutral.

        Args:
            query: The search query

        Returns:
            "time_sensitive", "historical", or "neutral"
        """
        query_lower = query.lower()

        # Check for time-sensitive keywords
        for keyword in self.TIME_SENSITIVE_KEYWORDS:
            if keyword in query_lower:
                return "time_sensitive"

        # Check for historical keywords
        for keyword in self.HISTORICAL_KEYWORDS:
            if keyword in query_lower:
                return "historical"

        return "neutral"

    def apply_adaptive_decay(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        Apply adaptive decay based on query type.

        Args:
            results: Retrieval results
            query: The search query

        Returns:
            Results with adaptive decay applied
        """
        query_type = self.classify_query(query)

        # Adjust half-life based on query type
        if query_type == "time_sensitive":
            half_life = self.sensitive_half_life
        elif query_type == "historical":
            half_life = self.historical_half_life
        else:
            half_life = self.config.half_life_days

        # Create temporary scorer with adjusted half-life
        temp_config = DecayConfig(
            half_life_days=half_life,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
            reference_date=self.reference_date,
            decay_type=self.config.decay_type
        )
        temp_scorer = TemporalDecayScorer(temp_config)

        return temp_scorer.apply_decay_to_results(results)


# Pre-configured scorers for common use cases
def create_aggressive_decay_scorer() -> TemporalDecayScorer:
    """
    Create scorer with aggressive decay (6-month half-life).
    Good for fast-changing domains.
    """
    return TemporalDecayScorer(DecayConfig(
        half_life_days=180,
        min_weight=0.05,
        decay_type="exponential"
    ))


def create_gentle_decay_scorer() -> TemporalDecayScorer:
    """
    Create scorer with gentle decay (2-year half-life).
    Good for stable domains with long-lived documents.
    """
    return TemporalDecayScorer(DecayConfig(
        half_life_days=730,
        min_weight=0.2,
        decay_type="exponential"
    ))


def create_step_decay_scorer() -> TemporalDecayScorer:
    """
    Create scorer with step decay.
    Good when you want clear recency tiers.
    """
    return TemporalDecayScorer(DecayConfig(
        decay_type="step",
        step_cutoffs=[
            (30, 1.0),      # Last month: 100%
            (90, 0.9),      # Last quarter: 90%
            (180, 0.7),     # Last 6 months: 70%
            (365, 0.5),     # Last year: 50%
            (730, 0.3),     # Last 2 years: 30%
        ],
        min_weight=0.1
    ))


# Convenience function
def apply_temporal_decay(
    results: List[RetrievalResult],
    half_life_days: int = 365
) -> List[RetrievalResult]:
    """
    Apply temporal decay to retrieval results.

    Args:
        results: List of retrieval results
        half_life_days: Decay half-life in days

    Returns:
        Results with decay applied, re-sorted
    """
    scorer = TemporalDecayScorer(DecayConfig(half_life_days=half_life_days))
    return scorer.apply_decay_to_results(results)
