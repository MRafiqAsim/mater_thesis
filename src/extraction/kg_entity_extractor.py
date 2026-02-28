"""
Knowledge Graph Entity Extractor Module

Modular extraction of entities for PathRAG knowledge graph construction.
Supports multiple strategies for benchmarking:
- SpaCyExtractor: Fast, local, rule-based NER
- LLMExtractor: LLM-based extraction (more accurate, slower)
- HybridExtractor: Combines both approaches

Usage:
    extractor = SpaCyKGExtractor(languages=["en", "nl"])
    entities = extractor.extract(text, language="en")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KGEntity:
    """A knowledge graph entity extracted from text"""
    text: str
    entity_type: str       # Original spaCy type (ORG, GPE, etc.)
    start: int
    end: int
    confidence: float = 0.85
    source: str = "spacy"  # Track extraction source for benchmarking
    pathrag_type: str = "" # PathRAG-compatible type (organization, geo, etc.)
    description: str = ""  # Entity description (for LLM extraction)
    is_pii: bool = False   # True if this entity should be anonymized
    anonymized_id: str = "" # Anonymized placeholder (e.g., "[PERSON_1]")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "type": self.entity_type,
            "pathrag_type": self.pathrag_type or self.entity_type.lower(),
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
        }
        if self.description:
            result["description"] = self.description
        if self.is_pii:
            result["is_pii"] = True
            if self.anonymized_id:
                result["anonymized_id"] = self.anonymized_id
        return result


# Entity types relevant for Knowledge Graph (PathRAG)
# PathRAG expects: ["organization", "person", "geo", "event", "category"]
# We map spaCy NER labels to PathRAG-compatible types

# spaCy entity types to extract
DEFAULT_KG_ENTITY_TYPES: Set[str] = {
    "ORG",         # Organizations → PathRAG "organization"
    "GPE",         # Geopolitical entities → PathRAG "geo"
    "LOC",         # Locations → PathRAG "geo"
    "FAC",         # Facilities → PathRAG "geo"
    "EVENT",       # Named events → PathRAG "event"
    "PRODUCT",     # Products (extra, useful for business context)
    "WORK_OF_ART", # Titles of documents, reports
    "LAW",         # Laws, regulations, standards
    "NORP",        # Nationalities, religions → could map to "category"
}

# Mapping from spaCy labels to PathRAG entity types
SPACY_TO_PATHRAG_TYPE: Dict[str, str] = {
    "ORG": "organization",
    "GPE": "geo",
    "LOC": "geo",
    "FAC": "geo",
    "EVENT": "event",
    "PRODUCT": "product",      # Extended type
    "WORK_OF_ART": "document", # Extended type
    "LAW": "regulation",       # Extended type
    "NORP": "category",
    "PERSON": "person",        # Tracked but anonymized
}

# Entity types that should be included for PathRAG even if they are PII
# These will be extracted but the actual text will be anonymized
PATHRAG_PII_ENTITY_TYPES: Set[str] = {
    "PERSON",  # PathRAG needs person entities for relationship tracking
}


class KGEntityExtractor(ABC):
    """
    Abstract base class for KG entity extractors.

    Implement this interface to create new extraction strategies
    for benchmarking different approaches.
    """

    @abstractmethod
    def extract(self, text: str, language: str = "en") -> List[KGEntity]:
        """
        Extract knowledge graph entities from text.

        Args:
            text: Text to extract entities from
            language: Language code (en, nl, etc.)

        Returns:
            List of KGEntity objects
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for logging/benchmarking"""
        pass


class SpaCyKGExtractor(KGEntityExtractor):
    """
    spaCy-based KG entity extractor.

    Fast, local extraction using spaCy NER models.
    Good for high-volume processing.
    """

    def __init__(
        self,
        languages: List[str] = None,
        entity_types: Set[str] = None,
        min_entity_length: int = 2,
        include_pii_entities: bool = True
    ):
        """
        Initialize spaCy extractor.

        Args:
            languages: Languages to load models for (default: ["en"])
            entity_types: Entity types to extract (default: DEFAULT_KG_ENTITY_TYPES)
            min_entity_length: Minimum entity text length
            include_pii_entities: Include PERSON entities for PathRAG (they will
                                  be marked for anonymization but tracked in KG)
        """
        self.languages = languages or ["en"]
        self.entity_types = entity_types or DEFAULT_KG_ENTITY_TYPES
        self.min_entity_length = min_entity_length
        self.include_pii_entities = include_pii_entities
        self.models: Dict[str, Any] = {}

        # If including PII entities for PathRAG, add PERSON to entity types
        if include_pii_entities:
            self.entity_types = self.entity_types | PATHRAG_PII_ENTITY_TYPES

        self._load_models()

    def _load_models(self) -> None:
        """Load spaCy models for configured languages"""
        import spacy

        model_map = {
            "en": "en_core_web_trf",
            "nl": "nl_core_news_lg",
            "de": "de_core_news_lg",
            "fr": "fr_core_news_lg",
        }

        for lang in self.languages:
            model_name = model_map.get(lang)
            if model_name:
                try:
                    self.models[lang] = spacy.load(model_name)
                    logger.info(f"SpaCyKGExtractor: Loaded {model_name}")
                except OSError:
                    logger.warning(f"SpaCyKGExtractor: Model {model_name} not found")

    def extract(self, text: str, language: str = "en") -> List[KGEntity]:
        """Extract KG entities using spaCy NER"""
        entities = []

        # Get model for language (fallback to English)
        nlp = self.models.get(language) or self.models.get("en")
        if not nlp:
            logger.warning(f"No spaCy model for language: {language}")
            return entities

        try:
            doc = nlp(text)
            seen: Set[tuple] = set()  # Deduplicate

            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    text_clean = ent.text.strip()
                    key = (text_clean.lower(), ent.label_)

                    if key not in seen and len(text_clean) >= self.min_entity_length:
                        seen.add(key)
                        # Map to PathRAG-compatible type
                        pathrag_type = SPACY_TO_PATHRAG_TYPE.get(ent.label_, ent.label_.lower())
                        # Mark PII entities (PERSON) - they will be anonymized
                        is_pii = ent.label_ in PATHRAG_PII_ENTITY_TYPES
                        entities.append(KGEntity(
                            text=text_clean,
                            entity_type=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.85,
                            source="spacy",
                            pathrag_type=pathrag_type,
                            is_pii=is_pii
                        ))

        except Exception as e:
            logger.error(f"SpaCy extraction error: {e}")

        return entities

    def get_supported_languages(self) -> List[str]:
        return list(self.models.keys())

    @property
    def name(self) -> str:
        return "SpaCyKGExtractor"


class LLMKGExtractor(KGEntityExtractor):
    """
    LLM-based KG entity extractor.

    Uses OpenAI/Azure OpenAI for more accurate entity extraction.
    Better for complex entities and relationships.
    Slower and requires API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        entity_types: Set[str] = None,
        # Azure OpenAI settings
        use_azure: bool = False,
        azure_endpoint: str = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str = None
    ):
        """
        Initialize LLM extractor.

        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model to use (for OpenAI) or deployment name (for Azure)
            entity_types: Entity types to extract
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name (defaults to model)
        """
        self.api_key = api_key
        self.model = model
        self.entity_types = entity_types or DEFAULT_KG_ENTITY_TYPES
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment or model

    def _get_client(self):
        """Get the appropriate OpenAI client"""
        if self.use_azure:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version
            )
        else:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)

    def extract(self, text: str, language: str = "en") -> List[KGEntity]:
        """Extract KG entities using LLM"""
        if not self.api_key:
            return []

        try:
            import json
            client = self._get_client()

            prompt = f"""Extract named entities from this text. Return JSON array with objects containing:
- "text": exact entity text
- "type": one of {list(self.entity_types)}
- "start": character start position
- "end": character end position

Only extract entities of the specified types. Be precise with positions.

Text:
{text[:3000]}

Return ONLY the JSON array, no other text."""

            # Use deployment name for Azure, model name for OpenAI
            model_name = self.azure_deployment if self.use_azure else self.model
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks (```json ... ```)
            if content.startswith("```"):
                # Extract JSON from code block
                import re
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group()

            # Parse JSON response
            if content.startswith("["):
                raw_entities = json.loads(content)
                return [
                    KGEntity(
                        text=e.get("text", ""),
                        entity_type=e.get("type", "UNKNOWN"),
                        start=e.get("start", 0),
                        end=e.get("end", 0),
                        confidence=0.95,
                        source="llm"
                    )
                    for e in raw_entities
                    if e.get("type") in self.entity_types
                ]

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")

        return []

    def get_supported_languages(self) -> List[str]:
        return ["en", "nl", "de", "fr", "es", "it"]  # LLM supports many languages

    @property
    def name(self) -> str:
        return f"LLMKGExtractor({self.model})"


class HybridKGExtractor(KGEntityExtractor):
    """
    Hybrid extractor combining spaCy and LLM.

    Uses spaCy for fast extraction, LLM for validation/enhancement.
    Good balance of speed and accuracy.
    """

    def __init__(
        self,
        spacy_extractor: SpaCyKGExtractor,
        llm_extractor: Optional[LLMKGExtractor] = None,
        use_llm_for_validation: bool = False
    ):
        """
        Initialize hybrid extractor.

        Args:
            spacy_extractor: Primary spaCy extractor
            llm_extractor: Optional LLM extractor for enhancement
            use_llm_for_validation: Whether to validate spaCy results with LLM
        """
        self.spacy_extractor = spacy_extractor
        self.llm_extractor = llm_extractor
        self.use_llm_for_validation = use_llm_for_validation

    def extract(self, text: str, language: str = "en") -> List[KGEntity]:
        """Extract using spaCy, optionally enhance with LLM"""
        # Primary extraction with spaCy
        entities = self.spacy_extractor.extract(text, language)

        # Optional LLM enhancement
        if self.llm_extractor and self.use_llm_for_validation:
            llm_entities = self.llm_extractor.extract(text, language)
            # Merge unique LLM entities
            seen = {(e.text.lower(), e.entity_type) for e in entities}
            for e in llm_entities:
                key = (e.text.lower(), e.entity_type)
                if key not in seen:
                    seen.add(key)
                    entities.append(e)

        return entities

    def get_supported_languages(self) -> List[str]:
        return self.spacy_extractor.get_supported_languages()

    @property
    def name(self) -> str:
        return "HybridKGExtractor"


# Factory function for easy creation
def create_kg_extractor(
    strategy: str = "spacy",
    languages: List[str] = None,
    openai_api_key: str = None,
    # Azure OpenAI settings
    use_azure: bool = False,
    azure_endpoint: str = None,
    azure_api_version: str = "2024-12-01-preview",
    azure_deployment: str = None,
    **kwargs
) -> KGEntityExtractor:
    """
    Factory function to create KG extractor.

    Args:
        strategy: "spacy", "llm", or "hybrid"
        languages: Languages to support
        openai_api_key: Required for LLM strategies
        use_azure: Whether to use Azure OpenAI
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_version: Azure API version
        azure_deployment: Azure deployment name

    Returns:
        Configured KGEntityExtractor instance
    """
    languages = languages or ["en"]

    if strategy == "spacy":
        return SpaCyKGExtractor(languages=languages, **kwargs)

    elif strategy == "llm":
        if not openai_api_key:
            raise ValueError("openai_api_key required for LLM strategy")
        return LLMKGExtractor(
            api_key=openai_api_key,
            use_azure=use_azure,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            **kwargs
        )

    elif strategy == "hybrid":
        spacy_ext = SpaCyKGExtractor(languages=languages)
        llm_ext = None
        if openai_api_key:
            llm_ext = LLMKGExtractor(
                api_key=openai_api_key,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment
            )
        return HybridKGExtractor(
            spacy_extractor=spacy_ext,
            llm_extractor=llm_ext,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
