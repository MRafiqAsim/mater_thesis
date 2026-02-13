"""
PII Detector

Detects Personally Identifiable Information (PII) in text using
multiple detection methods: Presidio, spaCy NER, and regex patterns.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    IBAN = "IBAN"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"                      # Social Security Number (US)
    BSN = "BSN"                      # Burger Service Nummer (NL)
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    ID_NUMBER = "ID_NUMBER"
    LICENSE_PLATE = "LICENSE_PLATE"
    PASSPORT = "PASSPORT"


@dataclass
class PIIEntity:
    """Detected PII entity"""

    text: str                        # The detected text
    pii_type: PIIType                # Type of PII
    start: int                       # Start character position
    end: int                         # End character position
    confidence: float                # Detection confidence (0-1)
    detection_method: str            # "presidio", "spacy", "regex"

    # For anonymization
    replacement: Optional[str] = None  # What to replace with

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.pii_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "method": self.detection_method,
        }


class PIIDetector:
    """
    Detect PII in text using multiple methods.

    Detection methods (in order of application):
    1. Microsoft Presidio - Primary detection engine
    2. spaCy NER - Named entity recognition for PERSON, ORG, LOCATION
    3. Regex patterns - Fallback for structured PII (emails, phones, etc.)

    Supports English and Dutch text.
    """

    # Regex patterns for structured PII
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        PIIType.PHONE: re.compile(
            # Phone numbers - captures common formats without trailing punctuation
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{1,4})?'
        ),
        PIIType.IBAN: re.compile(
            # IBAN - supports with/without spaces (e.g., NL91 ABNA 0417 1643 00)
            r'\b[A-Z]{2}\s?\d{2}\s?[A-Z0-9]{4}\s?\d{4}\s?\d{4}\s?\d{2,4}(?:\s?\d{2})?\b'
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        ),
        PIIType.IP_ADDRESS: re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ),
        PIIType.URL: re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*'
        ),
        PIIType.SSN: re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        ),
        PIIType.BSN: re.compile(
            r'\b\d{9}\b'  # Dutch BSN is 9 digits
        ),
        PIIType.LICENSE_PLATE: re.compile(
            # Dutch and Belgian formats
            r'\b[A-Z]{2}[-\s]?\d{2,3}[-\s]?[A-Z]{2}\b|\b\d[-\s]?[A-Z]{3}[-\s]?\d{3}\b'
        ),
    }

    # Entity types to detect
    # Note: DATE_OF_BIRTH removed - regular dates are not PII
    # Only actual birth dates with context (e.g., "DOB: 01/15/1985") should be PII
    DEFAULT_ENTITIES = [
        PIIType.PERSON,
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.IBAN,
        PIIType.ADDRESS,
        PIIType.CREDIT_CARD,
        PIIType.IP_ADDRESS,
    ]

    def __init__(
        self,
        entities: Optional[List[PIIType]] = None,
        languages: Optional[List[str]] = None,
        use_presidio: bool = True,
        use_spacy: bool = True,
        use_regex: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the PII detector.

        Args:
            entities: PII types to detect (default: common types)
            languages: Languages to support (default: ["en", "nl"])
            use_presidio: Use Microsoft Presidio
            use_spacy: Use spaCy NER
            use_regex: Use regex patterns
            confidence_threshold: Minimum confidence for detection
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.languages = languages or ["en", "nl"]
        self.use_presidio = use_presidio
        self.use_spacy = use_spacy
        self.use_regex = use_regex
        self.confidence_threshold = confidence_threshold

        # Initialize detection engines
        self._presidio_analyzer = None
        self._spacy_models = {}

        if use_presidio:
            self._init_presidio()

        if use_spacy:
            self._init_spacy()

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer"""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Configure NLP engine
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                    {"lang_code": "nl", "model_name": "nl_core_news_lg"},
                ],
            }

            try:
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
                self._presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            except Exception as e:
                logger.warning(f"Failed to load custom NLP config: {e}, using default")
                self._presidio_analyzer = AnalyzerEngine()

            logger.info("Presidio analyzer initialized")

        except ImportError:
            logger.warning("Presidio not available. Install: pip install presidio-analyzer presidio-anonymizer")
            self.use_presidio = False
        except Exception as e:
            # Catch other errors like Python version incompatibilities
            logger.warning(f"Presidio initialization failed: {e}")
            self.use_presidio = False

    def _init_spacy(self) -> None:
        """Initialize spaCy models"""
        try:
            import spacy

            model_map = {
                "en": "en_core_web_trf",  # or "en_core_web_lg" for faster processing
                "nl": "nl_core_news_lg",
            }

            for lang in self.languages:
                model_name = model_map.get(lang)
                if model_name:
                    try:
                        self._spacy_models[lang] = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model: {model_name}")
                    except OSError:
                        # Try to download
                        logger.info(f"Downloading spaCy model: {model_name}")
                        try:
                            spacy.cli.download(model_name)
                            self._spacy_models[lang] = spacy.load(model_name)
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")

        except ImportError:
            logger.warning("spaCy not available. Install: pip install spacy")
            self.use_spacy = False
        except Exception as e:
            # Catch other errors like Python version incompatibilities
            logger.warning(f"spaCy initialization failed: {e}")
            self.use_spacy = False

    def detect(
        self,
        text: str,
        language: str = "en"
    ) -> List[PIIEntity]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze
            language: Text language (en, nl)

        Returns:
            List of detected PII entities
        """
        if not text or not text.strip():
            return []

        all_entities = []

        # 1. Presidio detection
        if self.use_presidio and self._presidio_analyzer:
            presidio_entities = self._detect_with_presidio(text, language)
            all_entities.extend(presidio_entities)

        # 2. spaCy NER
        if self.use_spacy and language in self._spacy_models:
            spacy_entities = self._detect_with_spacy(text, language)
            all_entities.extend(spacy_entities)

        # 3. Regex patterns
        if self.use_regex:
            regex_entities = self._detect_with_regex(text)
            all_entities.extend(regex_entities)

        # Deduplicate and merge overlapping entities
        merged_entities = self._merge_entities(all_entities)

        # Filter by confidence
        filtered = [e for e in merged_entities
                   if e.confidence >= self.confidence_threshold]

        # Sort by position
        return sorted(filtered, key=lambda e: e.start)

    def _detect_with_presidio(
        self,
        text: str,
        language: str
    ) -> List[PIIEntity]:
        """Detect PII using Presidio"""
        entities = []

        # Map our PIIType to Presidio entity types
        presidio_entities = []
        for pii_type in self.entities:
            presidio_name = self._map_to_presidio(pii_type)
            if presidio_name:
                presidio_entities.append(presidio_name)

        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                entities=presidio_entities,
                language=language
            )

            for result in results:
                pii_type = self._map_from_presidio(result.entity_type)
                if pii_type:
                    entity = PIIEntity(
                        text=text[result.start:result.end],
                        pii_type=pii_type,
                        start=result.start,
                        end=result.end,
                        confidence=result.score,
                        detection_method="presidio"
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"Presidio detection failed: {e}")

        return entities

    def _detect_with_spacy(
        self,
        text: str,
        language: str
    ) -> List[PIIEntity]:
        """Detect named entities using spaCy"""
        entities = []

        nlp = self._spacy_models.get(language)
        if not nlp:
            return entities

        try:
            doc = nlp(text)

            # Map spaCy entity types to PII types
            spacy_mapping = {
                "PERSON": PIIType.PERSON,
                "PER": PIIType.PERSON,      # Dutch model
                "ORG": PIIType.ORGANIZATION,
                "GPE": PIIType.LOCATION,
                "LOC": PIIType.LOCATION,
            }

            for ent in doc.ents:
                pii_type = spacy_mapping.get(ent.label_)
                if pii_type and pii_type in self.entities:
                    entity = PIIEntity(
                        text=ent.text,
                        pii_type=pii_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.85,  # spaCy doesn't provide confidence
                        detection_method="spacy"
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"spaCy detection failed: {e}")

        return entities

    def _detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []

        for pii_type, pattern in self.PATTERNS.items():
            if pii_type not in self.entities:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()
                start = match.start()
                end = match.end()

                # Strip trailing punctuation for phone numbers
                if pii_type == PIIType.PHONE:
                    while matched_text and matched_text[-1] in '.,;:!?)':
                        matched_text = matched_text[:-1]
                        end -= 1

                # Validate the match
                if self._validate_regex_match(matched_text, pii_type):
                    entity = PIIEntity(
                        text=matched_text,
                        pii_type=pii_type,
                        start=start,
                        end=end,
                        confidence=0.9,  # High confidence for regex matches
                        detection_method="regex"
                    )
                    entities.append(entity)

        return entities

    def _validate_regex_match(self, text: str, pii_type: PIIType) -> bool:
        """Validate regex match to reduce false positives"""
        if pii_type == PIIType.PHONE:
            # Must have at least 7 digits
            digits = re.sub(r'\D', '', text)
            return len(digits) >= 7

        if pii_type == PIIType.IP_ADDRESS:
            # Validate IP address format
            parts = text.split('.')
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                return False

        if pii_type == PIIType.BSN:
            # Dutch BSN must pass 11-test
            return self._validate_bsn(text)

        return True

    def _validate_bsn(self, bsn: str) -> bool:
        """Validate Dutch BSN using 11-test"""
        bsn = re.sub(r'\D', '', bsn)
        if len(bsn) != 9:
            return False

        try:
            total = sum(int(bsn[i]) * (9 - i) for i in range(8))
            total -= int(bsn[8])
            return total % 11 == 0
        except (ValueError, IndexError):
            return False

    def _merge_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge overlapping entities, keeping highest confidence"""
        if not entities:
            return []

        # Sort by start position, then by confidence (desc)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.confidence))

        merged = []
        current = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if next_entity.start < current.end:
                # Overlapping - keep higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
                # If same type, extend the span
                elif next_entity.pii_type == current.pii_type:
                    current = PIIEntity(
                        text=current.text if current.end >= next_entity.end
                             else current.text + next_entity.text[current.end - next_entity.start:],
                        pii_type=current.pii_type,
                        start=current.start,
                        end=max(current.end, next_entity.end),
                        confidence=max(current.confidence, next_entity.confidence),
                        detection_method=current.detection_method
                    )
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _map_to_presidio(self, pii_type: PIIType) -> Optional[str]:
        """Map PIIType to Presidio entity type"""
        mapping = {
            PIIType.PERSON: "PERSON",
            PIIType.EMAIL: "EMAIL_ADDRESS",
            PIIType.PHONE: "PHONE_NUMBER",
            PIIType.ADDRESS: "ADDRESS",
            PIIType.IBAN: "IBAN_CODE",
            PIIType.CREDIT_CARD: "CREDIT_CARD",
            PIIType.SSN: "US_SSN",
            PIIType.IP_ADDRESS: "IP_ADDRESS",
            PIIType.URL: "URL",
            PIIType.LOCATION: "LOCATION",
            # PIIType.DATE_OF_BIRTH: "DATE_TIME",  # Disabled: regular dates are not PII
        }
        return mapping.get(pii_type)

    def _map_from_presidio(self, presidio_type: str) -> Optional[PIIType]:
        """Map Presidio entity type to PIIType"""
        mapping = {
            "PERSON": PIIType.PERSON,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "ADDRESS": PIIType.ADDRESS,
            "IBAN_CODE": PIIType.IBAN,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "US_SSN": PIIType.SSN,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "URL": PIIType.URL,
            "LOCATION": PIIType.LOCATION,
            # "DATE_TIME": PIIType.DATE_OF_BIRTH,  # Disabled: regular dates are not PII
            "NRP": PIIType.ID_NUMBER,  # National Registration Number
        }
        return mapping.get(presidio_type)


# Convenience function
def detect_pii(text: str, language: str = "en") -> List[PIIEntity]:
    """
    Detect PII in text.

    Args:
        text: Text to analyze
        language: Text language

    Returns:
        List of detected PII entities
    """
    detector = PIIDetector()
    return detector.detect(text, language)
