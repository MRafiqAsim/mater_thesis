"""
OpenAI-based PII Detector

Uses OpenAI API for PII detection with structured output.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .pii_detector import PIIEntity, PIIType

logger = logging.getLogger(__name__)


# System prompt for PII detection
PII_DETECTION_SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) detection expert.
Your task is to identify ALL PII entities in the given text.

PII types to detect:
- PERSON: Names of individuals (first name, last name, full name)
- EMAIL: Email addresses
- PHONE: Phone numbers (any format)
- ADDRESS: Physical addresses (street, city, postal code)
- IBAN: International Bank Account Numbers
- CREDIT_CARD: Credit/debit card numbers
- SSN: US Social Security Numbers
- BSN: Dutch Burger Service Nummer (9-digit Dutch ID)
- DATE_OF_BIRTH: Dates of birth
- IP_ADDRESS: IP addresses
- LOCATION: Geographic locations, cities, countries
- ORGANIZATION: Company/organization names
- ID_NUMBER: Any other identification numbers
- LICENSE_PLATE: Vehicle license plates
- PASSPORT: Passport numbers

IMPORTANT RULES:
1. Be thorough - identify ALL PII, even if partially visible
2. Include the exact text as it appears (preserve original formatting)
3. Provide accurate character positions (0-indexed)
4. Distinguish between person names and organization names
5. "John Deere" (the company) is ORGANIZATION, not PERSON
6. Generic emails like "info@company.com" are still EMAIL
7. Consider context - "Jan" in Dutch text is likely a PERSON name

Return your response as a JSON array of detected entities."""


PII_DETECTION_USER_PROMPT = """Analyze the following text and identify all PII entities.

Text:
\"\"\"
{text}
\"\"\"

Return a JSON array where each element has:
- "text": the exact PII text found
- "type": the PII type (PERSON, EMAIL, PHONE, etc.)
- "start": start character position (0-indexed)
- "end": end character position
- "confidence": your confidence score (0.0 to 1.0)
- "reasoning": brief explanation why this is PII

Example response:
[
  {{"text": "John Smith", "type": "PERSON", "start": 8, "end": 18, "confidence": 0.95, "reasoning": "Full person name"}},
  {{"text": "john@email.com", "type": "EMAIL", "start": 30, "end": 44, "confidence": 1.0, "reasoning": "Email address format"}}
]

If no PII is found, return an empty array: []"""


class OpenAIPIIDetector:
    """
    PII Detector using OpenAI API.

    Uses GPT models to identify PII with high accuracy,
    especially for context-dependent and ambiguous cases.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize OpenAI PII Detector.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.0 for deterministic)
            confidence_threshold: Minimum confidence to include entity
        """
        self.model = model
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI PII Detector initialized with model: {model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def detect(
        self,
        text: str,
        language: str = "en"
    ) -> List[PIIEntity]:
        """
        Detect PII in text using OpenAI.

        Args:
            text: Text to analyze
            language: Text language (used for context)

        Returns:
            List of detected PII entities
        """
        if not text or not text.strip():
            return []

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PII_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": PII_DETECTION_USER_PROMPT.format(text=text)}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Handle both array and object with "entities" key
            if isinstance(result, dict):
                entities_data = result.get("entities", result.get("pii_entities", []))
            else:
                entities_data = result

            # Convert to PIIEntity objects
            entities = []
            for item in entities_data:
                try:
                    pii_type = PIIType(item["type"])
                    confidence = float(item.get("confidence", 0.9))

                    if confidence >= self.confidence_threshold:
                        entity = PIIEntity(
                            text=item["text"],
                            pii_type=pii_type,
                            start=int(item["start"]),
                            end=int(item["end"]),
                            confidence=confidence,
                            detection_method="openai"
                        )
                        entities.append(entity)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse entity: {item}, error: {e}")
                    continue

            # Validate positions
            entities = self._validate_positions(entities, text)

            return sorted(entities, key=lambda e: e.start)

        except Exception as e:
            logger.error(f"OpenAI PII detection failed: {e}")
            return []

    def _validate_positions(
        self,
        entities: List[PIIEntity],
        text: str
    ) -> List[PIIEntity]:
        """Validate and fix entity positions"""
        validated = []

        for entity in entities:
            # Check if position is valid
            if entity.start < 0 or entity.end > len(text):
                # Try to find the text in the document
                idx = text.find(entity.text)
                if idx >= 0:
                    entity = PIIEntity(
                        text=entity.text,
                        pii_type=entity.pii_type,
                        start=idx,
                        end=idx + len(entity.text),
                        confidence=entity.confidence,
                        detection_method=entity.detection_method
                    )
                else:
                    logger.warning(f"Could not find entity text: {entity.text}")
                    continue

            # Verify the text matches
            actual_text = text[entity.start:entity.end]
            if actual_text != entity.text:
                # Try to find correct position
                idx = text.find(entity.text)
                if idx >= 0:
                    entity = PIIEntity(
                        text=entity.text,
                        pii_type=entity.pii_type,
                        start=idx,
                        end=idx + len(entity.text),
                        confidence=entity.confidence,
                        detection_method=entity.detection_method
                    )
                else:
                    # Use fuzzy match - the text might be slightly different
                    logger.debug(f"Text mismatch: expected '{entity.text}', found '{actual_text}'")

            validated.append(entity)

        return validated

    def detect_batch(
        self,
        texts: List[str],
        language: str = "en"
    ) -> List[List[PIIEntity]]:
        """
        Detect PII in multiple texts.

        Args:
            texts: List of texts to analyze
            language: Text language

        Returns:
            List of entity lists, one per input text
        """
        results = []
        for text in texts:
            entities = self.detect(text, language)
            results.append(entities)
        return results


class OpenAIAnonymizer:
    """
    Anonymizer using OpenAI API.

    Can generate realistic synthetic replacements for PII.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        generate_synthetic: bool = True
    ):
        """
        Initialize OpenAI Anonymizer.

        Args:
            api_key: OpenAI API key
            model: Model to use
            generate_synthetic: Generate realistic replacements
        """
        self.model = model
        self.generate_synthetic = generate_synthetic
        self._replacement_cache: Dict[str, str] = {}

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def anonymize(
        self,
        text: str,
        entities: List[PIIEntity],
        strategy: str = "replace"
    ) -> str:
        """
        Anonymize text by replacing PII entities.

        Args:
            text: Original text
            entities: Detected PII entities
            strategy: "replace" (placeholders), "synthetic" (realistic), "redact" (remove)

        Returns:
            Anonymized text
        """
        if not entities:
            return text

        # Sort entities by position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        result = text

        for entity in sorted_entities:
            if strategy == "redact":
                replacement = "[REDACTED]"
            elif strategy == "synthetic" and self.generate_synthetic:
                replacement = self._get_synthetic_replacement(entity)
            else:
                # Default placeholder replacement
                replacement = self._get_placeholder(entity)

            result = result[:entity.start] + replacement + result[entity.end:]

        return result

    def _get_placeholder(self, entity: PIIEntity) -> str:
        """Generate placeholder replacement"""
        type_name = entity.pii_type.value
        # Use hash for consistency
        hash_id = hash(entity.text) % 1000
        return f"[{type_name}_{hash_id}]"

    def _get_synthetic_replacement(self, entity: PIIEntity) -> str:
        """Generate synthetic realistic replacement"""
        # Check cache for consistency
        cache_key = f"{entity.pii_type.value}:{entity.text}"
        if cache_key in self._replacement_cache:
            return self._replacement_cache[cache_key]

        try:
            prompt = f"""Generate a realistic but fake replacement for this {entity.pii_type.value}:
Original: {entity.text}

Rules:
- Must be clearly fake but realistic format
- Same general type/format as original
- For PERSON names, use common but obviously fake names
- For EMAIL, use @example.com domain
- For PHONE, use 555 exchange (US) or similar fake numbers
- Keep similar length/format

Return ONLY the replacement text, nothing else."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50
            )

            replacement = response.choices[0].message.content.strip()
            self._replacement_cache[cache_key] = replacement
            return replacement

        except Exception as e:
            logger.warning(f"Failed to generate synthetic replacement: {e}")
            return self._get_placeholder(entity)

    def anonymize_with_detection(
        self,
        text: str,
        language: str = "en",
        strategy: str = "replace"
    ) -> tuple:
        """
        Detect PII and anonymize in one call.

        Returns:
            (anonymized_text, detected_entities)
        """
        detector = OpenAIPIIDetector(
            api_key=self.client.api_key,
            model=self.model
        )
        entities = detector.detect(text, language)
        anonymized = self.anonymize(text, entities, strategy)
        return anonymized, entities
