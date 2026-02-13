"""
Temporal Expression Extractor

Extracts temporal information from document content when metadata
is incomplete or unavailable. Uses pattern matching and NLP.
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from .models import TemporalSignals


class TemporalExpressionExtractor:
    """
    Extract temporal signals from text content.

    Used when document metadata doesn't contain reliable dates,
    or to supplement metadata with content-based signals.
    """

    # Explicit date patterns
    EXPLICIT_DATE_PATTERNS = [
        # ISO format: 2024-01-15
        r"\b(\d{4}-\d{2}-\d{2})\b",

        # US format: January 15, 2024 or Jan 15, 2024
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,.]?\s+\d{1,2}[,.]?\s+\d{4})\b",

        # EU format: 15 January 2024 or 15 Jan 2024
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,.]?\s+\d{4})\b",

        # Numeric: 01/15/2024 or 15/01/2024 or 15-01-2024
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",

        # Short year: 01/15/24
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2})\b",
    ]

    # Contextual date patterns (dates with context clues)
    CONTEXTUAL_DATE_PATTERNS = [
        (r"as of\s+(.+?)(?:\.|,|$)", "as_of"),
        (r"dated\s+(.+?)(?:\.|,|$)", "dated"),
        (r"effective\s+(.+?)(?:\.|,|$)", "effective"),
        (r"valid (?:from|until)\s+(.+?)(?:\.|,|$)", "validity"),
        (r"(?:updated|revised|amended)\s+(?:on\s+)?(.+?)(?:\.|,|$)", "updated"),
        (r"(?:created|written)\s+(?:on\s+)?(.+?)(?:\.|,|$)", "created"),
        (r"(?:sent|received)\s+(?:on\s+)?(.+?)(?:\.|,|$)", "sent"),
    ]

    # Quarter and fiscal year patterns
    QUARTER_PATTERNS = [
        r"\bQ([1-4])\s*[''']?\s*(\d{4})\b",           # Q1 2024, Q1'24
        r"\b(\d{4})\s*Q([1-4])\b",                     # 2024 Q1
        r"\b(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})\b",
    ]

    FISCAL_YEAR_PATTERNS = [
        r"\b(?:FY|fiscal\s+year)\s*[''']?\s*(\d{4})\b",  # FY2024, FY'24
        r"\b(?:FY|fiscal\s+year)\s*(\d{2})\b",           # FY24
    ]

    # Version indicators
    VERSION_PATTERNS = [
        r"\bversion\s*(\d+(?:\.\d+)*)\b",              # version 2.1
        r"\bv(\d+(?:\.\d+)*)\b",                        # v2.1
        r"\bdraft\s*(\d+)\b",                          # draft 3
        r"\brevision\s*(\d+)\b",                       # revision 5
        r"\brev[.]?\s*(\d+)\b",                        # rev. 2
        r"\bedition\s*(\d+)\b",                        # edition 3
        r"\brelease\s*(\d+(?:\.\d+)*)\b",              # release 1.2
    ]

    # Supersedes/replaces patterns
    SUPERSEDES_PATTERNS = [
        r"(?:this\s+)?(?:document\s+)?(?:supersedes|replaces|updates|amends)\s+(.+?)(?:\.|,|$)",
        r"(?:superseding|replacing|updating)\s+(.+?)(?:\.|,|$)",
        r"(?:obsoletes|cancels|voids)\s+(.+?)(?:\.|,|$)",
        r"(?:previous\s+version|earlier\s+version|old\s+version):\s*(.+?)(?:\.|,|$)",
    ]

    # Relative time expressions
    RELATIVE_TIME_PATTERNS = [
        r"\b(yesterday|today|tomorrow)\b",
        r"\b(last|next|this)\s+(week|month|year|quarter)\b",
        r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
        r"\b(earlier|later)\s+this\s+(week|month|year)\b",
    ]

    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize the extractor.

        Args:
            reference_date: Reference date for resolving relative expressions.
                          Defaults to current datetime.
        """
        self.reference_date = reference_date or datetime.now()

    def extract(self, text: str) -> TemporalSignals:
        """
        Extract all temporal signals from text.

        Args:
            text: The text to analyze

        Returns:
            TemporalSignals containing all extracted temporal information
        """
        signals = TemporalSignals()

        # Extract explicit dates
        explicit_dates, date_expressions = self._extract_explicit_dates(text)
        signals.explicit_dates = explicit_dates
        signals.date_expressions = date_expressions

        # Extract contextual dates
        contextual_dates = self._extract_contextual_dates(text)
        signals.explicit_dates.extend(contextual_dates)

        # Extract quarter/fiscal year references
        period_dates = self._extract_period_dates(text)
        signals.explicit_dates.extend(period_dates)

        # Extract version indicators
        signals.version_indicators = self._extract_versions(text)

        # Extract supersedes references
        signals.supersedes_references = self._extract_supersedes(text)

        # Extract relative time expressions
        signals.relative_references = self._extract_relative_time(text)

        # Deduplicate dates
        signals.explicit_dates = list(set(signals.explicit_dates))
        signals.explicit_dates.sort()

        # Infer best date
        signals.inferred_date = self._infer_best_date(signals)

        # Calculate confidence
        signals.confidence = self._calculate_confidence(signals)

        return signals

    def _extract_explicit_dates(
        self,
        text: str
    ) -> Tuple[List[datetime], List[str]]:
        """Extract explicit date mentions from text"""
        dates = []
        expressions = []

        for pattern in self.EXPLICIT_DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                expressions.append(match)
                try:
                    # Try to parse the date
                    parsed = date_parser.parse(match, fuzzy=True)
                    # Sanity check: date should be reasonable (1990-2030)
                    if 1990 <= parsed.year <= 2030:
                        dates.append(parsed)
                except (ValueError, TypeError):
                    pass

        return dates, expressions

    def _extract_contextual_dates(self, text: str) -> List[datetime]:
        """Extract dates with contextual clues (as of, dated, etc.)"""
        dates = []

        for pattern, context_type in self.CONTEXTUAL_DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse the matched date string
                    parsed = date_parser.parse(match, fuzzy=True)
                    if 1990 <= parsed.year <= 2030:
                        dates.append(parsed)
                except (ValueError, TypeError):
                    pass

        return dates

    def _extract_period_dates(self, text: str) -> List[datetime]:
        """Extract dates from quarter and fiscal year references"""
        dates = []

        # Quarter patterns
        for pattern in self.QUARTER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 2:
                        q, year = match
                        if not q.isdigit():
                            # Handle "first quarter 2024" format
                            quarter_map = {"first": 1, "second": 2, "third": 3, "fourth": 4}
                            q = str(quarter_map.get(q.lower(), 1))
                            year = match[0]
                        quarter = int(q)
                        year = int(year)
                        # Convert quarter to month (middle of quarter)
                        month = (quarter - 1) * 3 + 2  # Feb, May, Aug, Nov
                        dates.append(datetime(year, month, 15))
                except (ValueError, TypeError, IndexError):
                    pass

        # Fiscal year patterns
        for pattern in self.FISCAL_YEAR_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = int(match)
                    if year < 100:
                        year += 2000  # Convert 24 to 2024
                    # Use middle of fiscal year (assuming calendar year)
                    dates.append(datetime(year, 7, 1))
                except (ValueError, TypeError):
                    pass

        return dates

    def _extract_versions(self, text: str) -> List[str]:
        """Extract version indicators"""
        versions = []

        for pattern in self.VERSION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            versions.extend(matches)

        return list(set(versions))

    def _extract_supersedes(self, text: str) -> List[str]:
        """Extract references to superseded documents"""
        references = []

        for pattern in self.SUPERSEDES_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the reference
                ref = match.strip()
                if len(ref) > 5:  # Ignore very short matches
                    references.append(ref)

        return references

    def _extract_relative_time(self, text: str) -> List[str]:
        """Extract relative time expressions"""
        expressions = []

        for pattern in self.RELATIVE_TIME_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    expressions.append(" ".join(match))
                else:
                    expressions.append(match)

        return expressions

    def _infer_best_date(self, signals: TemporalSignals) -> Optional[datetime]:
        """Infer the most likely document date from signals"""
        if not signals.explicit_dates:
            return None

        # Prefer the most recent date as the "effective" date
        # (documents usually reference their own date most recently)
        return max(signals.explicit_dates)

    def _calculate_confidence(self, signals: TemporalSignals) -> float:
        """Calculate confidence in temporal inference"""
        score = 0.0

        # Explicit dates are strong signals
        if signals.explicit_dates:
            score += 0.4
            # Multiple consistent dates increase confidence
            if len(signals.explicit_dates) > 1:
                score += 0.1

        # Version indicators suggest intentional versioning
        if signals.version_indicators:
            score += 0.2

        # Supersedes references indicate document lineage
        if signals.supersedes_references:
            score += 0.2

        # Contextual clues (like "as of") are very reliable
        if signals.date_expressions:
            for expr in signals.date_expressions:
                if any(kw in expr.lower() for kw in ["as of", "dated", "effective"]):
                    score += 0.1
                    break

        return min(score, 1.0)

    def resolve_relative_date(
        self,
        expression: str,
        reference_date: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Resolve a relative date expression to an absolute date.

        Args:
            expression: Relative time expression (e.g., "2 weeks ago")
            reference_date: Reference point for resolution

        Returns:
            Resolved datetime or None if unable to resolve
        """
        ref = reference_date or self.reference_date
        expression = expression.lower().strip()

        # Simple expressions
        if expression == "yesterday":
            return ref - relativedelta(days=1)
        elif expression == "today":
            return ref
        elif expression == "tomorrow":
            return ref + relativedelta(days=1)

        # "last/next/this" expressions
        match = re.match(r"(last|next|this)\s+(week|month|year|quarter)", expression)
        if match:
            direction, unit = match.groups()
            if direction == "last":
                delta = -1
            elif direction == "next":
                delta = 1
            else:
                delta = 0

            if unit == "week":
                return ref + relativedelta(weeks=delta)
            elif unit == "month":
                return ref + relativedelta(months=delta)
            elif unit == "year":
                return ref + relativedelta(years=delta)
            elif unit == "quarter":
                return ref + relativedelta(months=delta * 3)

        # "N units ago" expressions
        match = re.match(r"(\d+)\s+(days?|weeks?|months?|years?)\s+ago", expression)
        if match:
            amount, unit = match.groups()
            amount = int(amount)
            unit = unit.rstrip("s")  # Normalize to singular

            if unit == "day":
                return ref - relativedelta(days=amount)
            elif unit == "week":
                return ref - relativedelta(weeks=amount)
            elif unit == "month":
                return ref - relativedelta(months=amount)
            elif unit == "year":
                return ref - relativedelta(years=amount)

        return None


# Convenience function for quick extraction
def extract_temporal_signals(text: str) -> TemporalSignals:
    """
    Extract temporal signals from text.

    Args:
        text: Text to analyze

    Returns:
        TemporalSignals object with extracted information
    """
    extractor = TemporalExpressionExtractor()
    return extractor.extract(text)
