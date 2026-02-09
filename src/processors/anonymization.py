"""
PII Anonymization Module
========================
Privacy-preserving anonymization using Microsoft Presidio.

Features:
- PII detection (names, emails, phone numbers, addresses, etc.)
- Global/International patterns (multi-country support)
- EU GDPR compliance patterns
- Pseudonymization with consistent mapping
- Reversible anonymization for authorized access
- Multilingual support (EN/NL + international)

Supports global enterprise data spanning multiple regions:
- Europe (EU VAT, IBAN, national IDs)
- North America (SSN, US phone, Canadian SIN)
- Asia-Pacific
- International formats

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """Detected PII entity."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    anonymized_text: Optional[str] = None


@dataclass
class AnonymizationResult:
    """Result from anonymization operation."""
    original_text: str
    anonymized_text: str
    pii_entities: List[PIIEntity]
    entity_counts: Dict[str, int]
    mapping: Dict[str, str]  # original -> anonymized


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization."""
    # PII types to detect (global/international)
    entities_to_detect: Set[str] = field(default_factory=lambda: {
        # Standard PII
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
        "DATE_TIME",
        "NRP",  # Nationality, religious, political group
        # Financial
        "IBAN_CODE",
        "CREDIT_CARD",
        "SWIFT_CODE",
        # Digital
        "IP_ADDRESS",
        "URL",
        # Government/Official IDs (International)
        "US_SSN",
        "PASSPORT_NUMBER",
        "NATIONAL_ID",
        "EU_VAT_NUMBER",
        # Enterprise
        "EMPLOYEE_ID",
        "COMPANY_REGISTRATION",
    })

    # Anonymization strategy
    strategy: str = "pseudonymize"  # replace, redact, hash, pseudonymize

    # Pseudonymization settings
    consistent_pseudonyms: bool = True  # Same input -> same output
    pseudonym_prefix: str = ""
    preserve_format: bool = True  # Keep similar format (e.g., email -> email-like)

    # Confidence threshold
    min_confidence: float = 0.7

    # Language
    language: str = "en"


class GlobalPIIRecognizer:
    """
    Custom recognizer for global/international PII patterns.

    Supports enterprise data spanning multiple regions:
    - Europe: IBAN (all countries), EU VAT numbers, national IDs
    - North America: SSN, US/Canada phone formats
    - International: Phone numbers, passport numbers, tax IDs
    """

    @staticmethod
    def get_recognizers():
        """Get Presidio recognizers for global patterns."""
        from presidio_analyzer import PatternRecognizer, Pattern

        recognizers = []

        # ============================================
        # INTERNATIONAL PHONE NUMBERS
        # ============================================
        # Generic international phone with country code
        intl_phone_pattern = Pattern(
            name="international_phone",
            regex=r"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{1,4}[\s.\-]?\d{1,4}[\s.\-]?\d{1,9}",
            score=0.75,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[intl_phone_pattern],
            supported_language="en",
        ))

        # US Phone: (xxx) xxx-xxxx or xxx-xxx-xxxx
        us_phone_pattern = Pattern(
            name="us_phone",
            regex=r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}",
            score=0.7,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[us_phone_pattern],
            supported_language="en",
        ))

        # ============================================
        # INTERNATIONAL IBAN (All Countries)
        # ============================================
        # Generic IBAN: 2 letter country + 2 check digits + up to 30 alphanumeric
        iban_pattern = Pattern(
            name="international_iban",
            regex=r"\b[A-Z]{2}\d{2}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{0,14}\b",
            score=0.9,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="IBAN_CODE",
            patterns=[iban_pattern],
            supported_language="en",
        ))

        # ============================================
        # EU VAT NUMBERS (Multiple Countries)
        # ============================================
        # Format: 2-letter country code + 8-12 digits/chars
        eu_vat_pattern = Pattern(
            name="eu_vat",
            regex=r"\b(AT|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK|GB|XI)[A-Z0-9]{8,12}\b",
            score=0.85,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="EU_VAT_NUMBER",
            patterns=[eu_vat_pattern],
            supported_language="en",
        ))

        # ============================================
        # US SOCIAL SECURITY NUMBER
        # ============================================
        ssn_pattern = Pattern(
            name="us_ssn",
            regex=r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b",
            score=0.8,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[ssn_pattern],
            supported_language="en",
        ))

        # ============================================
        # PASSPORT NUMBERS (Generic)
        # ============================================
        passport_pattern = Pattern(
            name="passport_number",
            regex=r"\b[A-Z]{1,2}\d{6,9}\b",
            score=0.6,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="PASSPORT_NUMBER",
            patterns=[passport_pattern],
            supported_language="en",
        ))

        # ============================================
        # EMPLOYEE/STAFF IDs (Common Enterprise Patterns)
        # ============================================
        employee_id_pattern = Pattern(
            name="employee_id",
            regex=r"\b(EMP|STAFF|ID|USR)[\-_]?\d{4,8}\b",
            score=0.7,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="EMPLOYEE_ID",
            patterns=[employee_id_pattern],
            supported_language="en",
        ))

        # ============================================
        # SWIFT/BIC CODES (International Banking)
        # ============================================
        swift_pattern = Pattern(
            name="swift_bic",
            regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
            score=0.8,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="SWIFT_CODE",
            patterns=[swift_pattern],
            supported_language="en",
        ))

        # ============================================
        # GENERIC NATIONAL ID (Numeric patterns)
        # ============================================
        # Catches various national ID formats: XX.XX.XX-XXX.XX, XXX-XX-XXXX, etc.
        national_id_pattern = Pattern(
            name="national_id_generic",
            regex=r"\b\d{2,3}[\.\-/]\d{2,3}[\.\-/]\d{2,4}([\.\-/]\d{2,4})?\b",
            score=0.65,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[national_id_pattern],
            supported_language="en",
        ))

        # ============================================
        # COMPANY REGISTRATION NUMBERS
        # ============================================
        company_reg_pattern = Pattern(
            name="company_registration",
            regex=r"\b(REG|CRN|KVK|SIREN|SIRET)[\s\-:]?\d{6,14}\b",
            score=0.75,
        )
        recognizers.append(PatternRecognizer(
            supported_entity="COMPANY_REGISTRATION",
            patterns=[company_reg_pattern],
            supported_language="en",
        ))

        return recognizers


class PseudonymGenerator:
    """
    Generate consistent pseudonyms for PII entities.

    Uses deterministic hashing for consistency across documents.
    """

    def __init__(self, seed: str = "thesis_2024"):
        self.seed = seed
        self.mapping: Dict[str, str] = {}
        self.reverse_mapping: Dict[str, str] = {}

        # Pseudonym templates (global/international)
        self.templates = {
            "PERSON": self._generate_person_pseudonym,
            "EMAIL_ADDRESS": self._generate_email_pseudonym,
            "PHONE_NUMBER": self._generate_phone_pseudonym,
            "LOCATION": self._generate_location_pseudonym,
            "IBAN_CODE": self._generate_iban_pseudonym,
            "CREDIT_CARD": self._generate_card_pseudonym,
            "IP_ADDRESS": self._generate_ip_pseudonym,
            # International IDs
            "US_SSN": self._generate_ssn_pseudonym,
            "PASSPORT_NUMBER": self._generate_passport_pseudonym,
            "NATIONAL_ID": self._generate_national_id_pseudonym,
            "EU_VAT_NUMBER": self._generate_vat_pseudonym,
            # Enterprise
            "EMPLOYEE_ID": self._generate_employee_id_pseudonym,
            "COMPANY_REGISTRATION": self._generate_company_reg_pseudonym,
            "SWIFT_CODE": self._generate_swift_pseudonym,
        }

        # Counters for unique pseudonyms
        self.counters: Dict[str, int] = defaultdict(int)

    def _hash_value(self, value: str) -> str:
        """Create deterministic hash."""
        return hashlib.sha256(f"{self.seed}:{value}".encode()).hexdigest()[:8]

    def get_pseudonym(
        self,
        original: str,
        entity_type: str,
        preserve_format: bool = True
    ) -> str:
        """
        Get or create pseudonym for value.

        Args:
            original: Original PII value
            entity_type: Type of PII
            preserve_format: Whether to preserve similar format

        Returns:
            Pseudonymized value
        """
        # Check existing mapping
        key = f"{entity_type}:{original}"
        if key in self.mapping:
            return self.mapping[key]

        # Generate new pseudonym
        if preserve_format and entity_type in self.templates:
            pseudonym = self.templates[entity_type](original)
        else:
            pseudonym = f"[{entity_type}_{self._hash_value(original)}]"

        # Store mapping
        self.mapping[key] = pseudonym
        self.reverse_mapping[pseudonym] = original

        return pseudonym

    def _generate_person_pseudonym(self, original: str) -> str:
        """Generate person name pseudonym."""
        self.counters["PERSON"] += 1
        hash_val = self._hash_value(original)[:4].upper()
        return f"Person_{hash_val}"

    def _generate_email_pseudonym(self, original: str) -> str:
        """Generate email pseudonym preserving format."""
        hash_val = self._hash_value(original)[:6]
        return f"user_{hash_val}@anonymized.local"

    def _generate_phone_pseudonym(self, original: str) -> str:
        """Generate phone pseudonym preserving format hint."""
        hash_val = self._hash_value(original)
        # Preserve country code pattern if present
        if original.startswith("+"):
            # Extract country code length (1-3 digits)
            return f"+XX-XXX-XXX-{hash_val[:4]}"
        return f"XXX-XXX-{hash_val[:4]}"

    def _generate_location_pseudonym(self, original: str) -> str:
        """Generate location pseudonym."""
        hash_val = self._hash_value(original)[:4].upper()
        return f"Location_{hash_val}"

    def _generate_iban_pseudonym(self, original: str) -> str:
        """Generate IBAN pseudonym preserving country code pattern."""
        # Preserve country code if detectable
        if len(original) >= 2 and original[:2].isalpha():
            country = original[:2].upper()
            return f"{country}XX XXXX XXXX XXXX XXXX"
        return "XXXX XXXX XXXX XXXX XXXX"

    def _generate_vat_pseudonym(self, original: str) -> str:
        """Generate VAT pseudonym preserving country code."""
        if len(original) >= 2 and original[:2].isalpha():
            country = original[:2].upper()
            return f"{country}XXXXXXXXXX"
        return "XXXXXXXXXXXX"

    def _generate_card_pseudonym(self, original: str) -> str:
        """Generate credit card pseudonym."""
        return "XXXX-XXXX-XXXX-XXXX"

    def _generate_ip_pseudonym(self, original: str) -> str:
        """Generate IP address pseudonym."""
        hash_val = self._hash_value(original)
        return f"10.0.{int(hash_val[:2], 16) % 256}.{int(hash_val[2:4], 16) % 256}"

    def _generate_ssn_pseudonym(self, original: str) -> str:
        """Generate US SSN pseudonym."""
        return "XXX-XX-XXXX"

    def _generate_passport_pseudonym(self, original: str) -> str:
        """Generate passport number pseudonym."""
        hash_val = self._hash_value(original)
        return f"XX{hash_val[:6].upper()}"

    def _generate_national_id_pseudonym(self, original: str) -> str:
        """Generate generic national ID pseudonym."""
        return "XX-XXXXXX-XX"

    def _generate_employee_id_pseudonym(self, original: str) -> str:
        """Generate employee ID pseudonym."""
        hash_val = self._hash_value(original)
        return f"EMP-{hash_val[:6].upper()}"

    def _generate_company_reg_pseudonym(self, original: str) -> str:
        """Generate company registration pseudonym."""
        return "REG-XXXXXXXX"

    def _generate_swift_pseudonym(self, original: str) -> str:
        """Generate SWIFT/BIC code pseudonym."""
        return "XXXXXX2AXXX"

    def get_mapping_json(self) -> str:
        """Export mapping as JSON for secure storage."""
        return json.dumps(self.mapping, indent=2)

    def load_mapping(self, mapping_json: str):
        """Load existing mapping."""
        self.mapping = json.loads(mapping_json)
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}


class PIIAnonymizer:
    """
    Main PII anonymization class using Microsoft Presidio.

    Usage:
        anonymizer = PIIAnonymizer()
        result = anonymizer.anonymize("John Doe's email is john@example.com")
        print(result.anonymized_text)
    """

    def __init__(self, config: Optional[AnonymizationConfig] = None):
        """
        Initialize anonymizer.

        Args:
            config: Anonymization configuration
        """
        self.config = config or AnonymizationConfig()
        self.pseudonym_generator = PseudonymGenerator()
        self._setup_presidio()

    def _setup_presidio(self):
        """Initialize Presidio analyzer and anonymizer."""
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        # Setup registry with custom recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        # Add global/international recognizers
        for recognizer in GlobalPIIRecognizer.get_recognizers():
            registry.add_recognizer(recognizer)

        # Create analyzer
        self.analyzer = AnalyzerEngine(registry=registry)

        # Create anonymizer
        self.anonymizer_engine = AnonymizerEngine()

    def detect_pii(
        self,
        text: str,
        language: Optional[str] = None
    ) -> List[PIIEntity]:
        """
        Detect PII entities in text.

        Args:
            text: Input text
            language: Language code (default from config)

        Returns:
            List of detected PII entities
        """
        language = language or self.config.language

        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=list(self.config.entities_to_detect),
            score_threshold=self.config.min_confidence,
        )

        entities = []
        for result in results:
            entities.append(PIIEntity(
                entity_type=result.entity_type,
                text=text[result.start:result.end],
                start=result.start,
                end=result.end,
                score=result.score,
            ))

        return entities

    def anonymize(
        self,
        text: str,
        language: Optional[str] = None
    ) -> AnonymizationResult:
        """
        Anonymize text by replacing PII.

        Args:
            text: Input text
            language: Language code

        Returns:
            AnonymizationResult with anonymized text and metadata
        """
        language = language or self.config.language

        # Detect PII
        pii_entities = self.detect_pii(text, language)

        if not pii_entities:
            return AnonymizationResult(
                original_text=text,
                anonymized_text=text,
                pii_entities=[],
                entity_counts={},
                mapping={},
            )

        # Sort by position (reverse for replacement)
        pii_entities.sort(key=lambda x: x.start, reverse=True)

        # Apply anonymization
        anonymized_text = text
        mapping = {}
        entity_counts = defaultdict(int)

        for entity in pii_entities:
            if self.config.strategy == "pseudonymize":
                replacement = self.pseudonym_generator.get_pseudonym(
                    entity.text,
                    entity.entity_type,
                    self.config.preserve_format,
                )
            elif self.config.strategy == "redact":
                replacement = f"[{entity.entity_type}]"
            elif self.config.strategy == "hash":
                replacement = f"#{hashlib.md5(entity.text.encode()).hexdigest()[:8]}#"
            else:  # replace
                replacement = f"<{entity.entity_type}>"

            entity.anonymized_text = replacement
            mapping[entity.text] = replacement
            entity_counts[entity.entity_type] += 1

            # Replace in text
            anonymized_text = (
                anonymized_text[:entity.start] +
                replacement +
                anonymized_text[entity.end:]
            )

        # Re-sort entities by original position for output
        pii_entities.sort(key=lambda x: x.start)

        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized_text,
            pii_entities=pii_entities,
            entity_counts=dict(entity_counts),
            mapping=mapping,
        )

    def anonymize_batch(
        self,
        texts: List[str],
        language: Optional[str] = None
    ) -> List[AnonymizationResult]:
        """Anonymize multiple texts."""
        return [self.anonymize(text, language) for text in texts]

    def get_pii_statistics(
        self,
        results: List[AnonymizationResult]
    ) -> Dict[str, Any]:
        """
        Get aggregate PII statistics.

        Args:
            results: List of anonymization results

        Returns:
            Statistics dictionary
        """
        total_pii = 0
        type_counts = defaultdict(int)
        unique_values = defaultdict(set)

        for result in results:
            total_pii += len(result.pii_entities)
            for entity in result.pii_entities:
                type_counts[entity.entity_type] += 1
                unique_values[entity.entity_type].add(entity.text)

        return {
            "total_pii_detected": total_pii,
            "total_documents": len(results),
            "pii_per_document": total_pii / len(results) if results else 0,
            "by_type": {
                entity_type: {
                    "count": count,
                    "unique": len(unique_values[entity_type]),
                }
                for entity_type, count in type_counts.items()
            },
        }

    def export_mapping(self) -> str:
        """Export pseudonym mapping for secure storage."""
        return self.pseudonym_generator.get_mapping_json()

    def import_mapping(self, mapping_json: str):
        """Import existing pseudonym mapping."""
        self.pseudonym_generator.load_mapping(mapping_json)


# Export
__all__ = [
    'PIIAnonymizer',
    'PIIEntity',
    'AnonymizationResult',
    'AnonymizationConfig',
    'PseudonymGenerator',
    'GlobalPIIRecognizer',
]
