"""
Unified Processor

Combines PII detection, anonymization, and summarization based on configuration.
Supports three modes: OPENAI, LOCAL, and HYBRID.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, ProcessingMode, get_config
from .pii_detector import PIIDetector, PIIEntity, PIIType

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of unified text processing"""
    original_text: str
    anonymized_text: str
    summary: Optional[str]
    pii_entities: List[PIIEntity]
    key_entities: List[str]
    key_topics: List[str]
    processing_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "summary": self.summary,
            "pii_entities": [e.to_dict() for e in self.pii_entities],
            "pii_count": len(self.pii_entities),
            "key_entities": self.key_entities,
            "key_topics": self.key_topics,
            "processing_mode": self.processing_mode,
            "metadata": self.metadata,
        }


class UnifiedProcessor:
    """
    Unified text processor that handles PII detection, anonymization, and summarization.

    Supports three modes based on configuration:
    - OPENAI: Use OpenAI API for all processing
    - LOCAL: Use local models (Presidio/spaCy/regex)
    - HYBRID: Combine local for high-confidence, OpenAI for complex cases
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the unified processor.

        Args:
            config: Pipeline configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        self._local_detector = None
        self._local_anonymizer = None
        self._openai_detector = None
        self._openai_anonymizer = None
        self._openai_summarizer = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize processing components based on mode"""
        mode = self.config.mode

        if mode in [ProcessingMode.LOCAL, ProcessingMode.HYBRID]:
            self._init_local_components()

        if mode in [ProcessingMode.OPENAI, ProcessingMode.HYBRID]:
            self._init_openai_components()

    def _init_local_components(self):
        """Initialize local processing components"""
        try:
            from .pii_detector import PIIDetector
            from .anonymizer import Anonymizer

            self._local_detector = PIIDetector(
                confidence_threshold=self.config.pii.confidence_threshold,
                use_presidio=self.config.pii.use_presidio,
                use_spacy=self.config.pii.use_spacy,
                use_regex=self.config.pii.use_regex,
            )
            self._local_anonymizer = Anonymizer()
            logger.info("Local processing components initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize local components: {e}")
            if self.config.mode == ProcessingMode.LOCAL:
                raise

    def _init_openai_components(self):
        """Initialize OpenAI processing components"""
        try:
            from .openai_pii_detector import OpenAIPIIDetector, OpenAIAnonymizer
            from .openai_summarizer import OpenAISummarizer

            api_key = self.config.openai.api_key
            model = self.config.openai.model

            if not api_key:
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

            self._openai_detector = OpenAIPIIDetector(
                api_key=api_key,
                model=model,
                confidence_threshold=self.config.pii.confidence_threshold,
            )
            self._openai_anonymizer = OpenAIAnonymizer(
                api_key=api_key,
                model=model,
                generate_synthetic=self.config.anonymization.generate_synthetic,
            )
            self._openai_summarizer = OpenAISummarizer(
                api_key=api_key,
                model=model,
                max_summary_length=self.config.summarization.max_summary_length,
            )
            logger.info("OpenAI processing components initialized")

        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI components: {e}")
            raise

    def process(
        self,
        text: str,
        language: str = "en",
        include_summary: bool = True
    ) -> ProcessingResult:
        """
        Process text: detect PII, anonymize, and optionally summarize.

        Args:
            text: Text to process
            language: Text language
            include_summary: Whether to generate summary

        Returns:
            ProcessingResult with all outputs
        """
        if not text or not text.strip():
            return ProcessingResult(
                original_text=text,
                anonymized_text=text,
                summary=None,
                pii_entities=[],
                key_entities=[],
                key_topics=[],
                processing_mode=self.config.mode.value,
            )

        mode = self.config.mode

        if mode == ProcessingMode.OPENAI:
            return self._process_openai(text, language, include_summary)
        elif mode == ProcessingMode.LOCAL:
            return self._process_local(text, language, include_summary)
        else:  # HYBRID
            return self._process_hybrid(text, language, include_summary)

    def _process_openai(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using OpenAI API"""
        # Detect PII
        pii_entities = self._openai_detector.detect(text, language)

        # Anonymize
        strategy = self.config.anonymization.default_strategy
        if strategy == "synthetic":
            anonymized = self._openai_anonymizer.anonymize(
                text, pii_entities, strategy="synthetic"
            )
        else:
            anonymized = self._openai_anonymizer.anonymize(
                text, pii_entities, strategy="replace"
            )

        # Summarize (on anonymized text for privacy)
        summary = None
        key_entities = []
        key_topics = []

        if include_summary and self.config.summarization.enabled:
            summary_result = self._openai_summarizer.summarize(
                anonymized,
                style=self.config.summarization.summary_style,
            )
            summary = summary_result.summary
            key_entities = summary_result.key_entities
            key_topics = summary_result.key_topics

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=pii_entities,
            key_entities=key_entities,
            key_topics=key_topics,
            processing_mode="openai",
        )

    def _process_local(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using local models"""
        # Detect PII
        pii_entities = self._local_detector.detect(text, language)

        # Anonymize - use keyword arguments to avoid position errors
        from .anonymizer import AnonymizationStrategy
        strategy_map = {
            "replace": AnonymizationStrategy.REPLACE,
            "hash": AnonymizationStrategy.HASH,
            "mask": AnonymizationStrategy.MASK,
            "redact": AnonymizationStrategy.REDACT,
        }
        strategy = strategy_map.get(
            self.config.anonymization.default_strategy,
            AnonymizationStrategy.REPLACE
        )

        result = self._local_anonymizer.anonymize(
            text=text,
            language=language,
            strategy=strategy,
            entities=pii_entities,
        )
        anonymized = result.anonymized_text

        # No local summarization available
        summary = None
        if include_summary:
            logger.warning("Summarization not available in LOCAL mode")

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=pii_entities,
            key_entities=[],
            key_topics=[],
            processing_mode="local",
        )

    def _process_hybrid(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using hybrid approach: local first, OpenAI for uncertain cases"""
        # Step 1: Local detection
        local_entities = self._local_detector.detect(text, language)

        # Step 2: Separate high and low confidence entities
        high_conf_entities = []
        low_conf_entities = []
        threshold = self.config.pii.hybrid_confidence_threshold

        for entity in local_entities:
            if entity.confidence >= threshold:
                high_conf_entities.append(entity)
            else:
                low_conf_entities.append(entity)

        # Step 3: Use OpenAI for verification if there are low-confidence entities
        # or if we suspect there might be missed entities
        final_entities = high_conf_entities.copy()

        if low_conf_entities or self._should_verify_with_openai(text, local_entities):
            openai_entities = self._openai_detector.detect(text, language)

            # Merge: keep OpenAI entities that don't overlap with high-conf local
            for oe in openai_entities:
                overlap = False
                for le in high_conf_entities:
                    if self._entities_overlap(oe, le):
                        overlap = True
                        break
                if not overlap:
                    final_entities.append(oe)

        # Deduplicate
        final_entities = self._deduplicate_entities(final_entities)

        # Anonymize using local anonymizer - use keyword arguments
        from .anonymizer import AnonymizationStrategy
        strategy_map = {
            "replace": AnonymizationStrategy.REPLACE,
            "hash": AnonymizationStrategy.HASH,
            "mask": AnonymizationStrategy.MASK,
            "redact": AnonymizationStrategy.REDACT,
        }
        strategy = strategy_map.get(
            self.config.anonymization.default_strategy,
            AnonymizationStrategy.REPLACE
        )

        result = self._local_anonymizer.anonymize(
            text=text,
            language=language,
            strategy=strategy,
            entities=final_entities,
        )
        anonymized = result.anonymized_text

        # Summarize with OpenAI
        summary = None
        key_entities = []
        key_topics = []

        if include_summary and self.config.summarization.enabled:
            summary_result = self._openai_summarizer.summarize(
                anonymized,
                style=self.config.summarization.summary_style,
            )
            summary = summary_result.summary
            key_entities = summary_result.key_entities
            key_topics = summary_result.key_topics

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=final_entities,
            key_entities=key_entities,
            key_topics=key_topics,
            processing_mode="hybrid",
            metadata={
                "local_entities_count": len(local_entities),
                "high_conf_count": len(high_conf_entities),
                "openai_verified": True,
            }
        )

    def _should_verify_with_openai(
        self,
        text: str,
        local_entities: List[PIIEntity]
    ) -> bool:
        """Determine if OpenAI verification is needed"""
        # Verify if:
        # 1. Text is long but few entities detected
        # 2. Text contains patterns that might indicate missed PII
        # 3. No PERSON entities detected in text with names-like patterns

        words = text.split()
        if len(words) > 50 and len(local_entities) == 0:
            return True

        # Check for potential missed person names (capitalized words not at sentence start)
        import re
        potential_names = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', text)
        person_entities = [e for e in local_entities if e.pii_type == PIIType.PERSON]
        if len(potential_names) > len(person_entities) * 2:
            return True

        return False

    def _entities_overlap(self, e1: PIIEntity, e2: PIIEntity) -> bool:
        """Check if two entities overlap"""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    def _deduplicate_entities(
        self,
        entities: List[PIIEntity]
    ) -> List[PIIEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        if not entities:
            return []

        # Sort by position, then confidence
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -e.confidence)
        )

        result = []
        for entity in sorted_entities:
            # Check if overlaps with any existing
            overlap = False
            for existing in result:
                if self._entities_overlap(entity, existing):
                    overlap = True
                    break
            if not overlap:
                result.append(entity)

        return sorted(result, key=lambda e: e.start)

    def process_batch(
        self,
        texts: List[str],
        language: str = "en",
        include_summary: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple texts.

        Args:
            texts: List of texts to process
            language: Text language
            include_summary: Whether to generate summaries

        Returns:
            List of ProcessingResult objects
        """
        return [
            self.process(text, language, include_summary)
            for text in texts
        ]


# Convenience function
def process_text(
    text: str,
    language: str = "en",
    mode: str = "openai",
    include_summary: bool = True
) -> ProcessingResult:
    """
    Process text with specified mode.

    Args:
        text: Text to process
        language: Text language
        mode: Processing mode ("openai", "local", "hybrid")
        include_summary: Whether to generate summary

    Returns:
        ProcessingResult
    """
    from config import init_config

    config = init_config(mode=mode)
    processor = UnifiedProcessor(config)
    return processor.process(text, language, include_summary)
