"""
Named Entity Recognition (NER) Module
=====================================
Multilingual NER extraction using spaCy for English and Dutch.

Entity Types:
- PERSON - People names
- ORG - Organizations, companies
- GPE - Geopolitical entities (countries, cities)
- DATE - Dates and time expressions
- MONEY - Monetary values
- PROJECT - Project names (custom)
- TECHNOLOGY - Technology terms (custom)

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted named entity."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    normalized_text: Optional[str] = None
    source_language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "normalized_text": self.normalized_text or self.text,
            "source_language": self.source_language,
        }


@dataclass
class NERResult:
    """Result from NER extraction."""
    entities: List[Entity]
    entity_counts: Dict[str, int]
    text_length: int
    language: str
    model_used: str


@dataclass
class NERConfig:
    """Configuration for NER extraction."""
    # spaCy models
    english_model: str = "en_core_web_trf"  # Transformer-based for accuracy
    dutch_model: str = "nl_core_news_lg"    # Large Dutch model

    # Fallback models (smaller, faster)
    english_model_fallback: str = "en_core_web_sm"
    dutch_model_fallback: str = "nl_core_news_sm"

    # Entity types to extract
    entity_types: Set[str] = field(default_factory=lambda: {
        "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
        "MONEY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "FAC", "NORP"
    })

    # Custom entity patterns
    enable_custom_patterns: bool = True

    # Normalization
    normalize_entities: bool = True
    lowercase_normalize: bool = False


class MultilingualNER:
    """
    Multilingual Named Entity Recognition using spaCy.

    Supports English and Dutch with automatic language detection
    and model selection.
    """

    def __init__(self, config: Optional[NERConfig] = None):
        """
        Initialize NER with language models.

        Args:
            config: NER configuration
        """
        self.config = config or NERConfig()
        self.models = {}
        self._load_models()

        if self.config.enable_custom_patterns:
            self._setup_custom_patterns()

    def _load_models(self):
        """Load spaCy models for each language."""
        import spacy

        # Load English model
        try:
            self.models["en"] = spacy.load(self.config.english_model)
            logger.info(f"Loaded English model: {self.config.english_model}")
        except OSError:
            try:
                self.models["en"] = spacy.load(self.config.english_model_fallback)
                logger.warning(f"Using fallback English model: {self.config.english_model_fallback}")
            except OSError:
                logger.error("No English spaCy model available. Install with: python -m spacy download en_core_web_sm")

        # Load Dutch model
        try:
            self.models["nl"] = spacy.load(self.config.dutch_model)
            logger.info(f"Loaded Dutch model: {self.config.dutch_model}")
        except OSError:
            try:
                self.models["nl"] = spacy.load(self.config.dutch_model_fallback)
                logger.warning(f"Using fallback Dutch model: {self.config.dutch_model_fallback}")
            except OSError:
                logger.error("No Dutch spaCy model available. Install with: python -m spacy download nl_core_news_sm")

    def _setup_custom_patterns(self):
        """Setup custom entity patterns using EntityRuler."""
        from spacy.pipeline import EntityRuler

        # Custom patterns for domain-specific entities
        custom_patterns = [
            # Project patterns
            {"label": "PROJECT", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}-\d{3,6}$"}}]},
            {"label": "PROJECT", "pattern": [{"LOWER": "project"}, {"IS_TITLE": True}]},

            # Technology patterns
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": [
                "azure", "aws", "python", "java", "kubernetes", "docker",
                "tensorflow", "pytorch", "spark", "databricks", "openai",
                "langchain", "gpt", "llm", "api", "sql", "nosql"
            ]}}]},

            # Document reference patterns
            {"label": "DOCUMENT_REF", "pattern": [{"TEXT": {"REGEX": r"^DOC-\d{4,8}$"}}]},

            # Email patterns (backup)
            {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"^[\w\.-]+@[\w\.-]+\.\w+$"}}]},
        ]

        for lang, model in self.models.items():
            if "entity_ruler" not in model.pipe_names:
                ruler = model.add_pipe("entity_ruler", before="ner")
                ruler.add_patterns(custom_patterns)

    def extract(
        self,
        text: str,
        language: str = "en"
    ) -> NERResult:
        """
        Extract named entities from text.

        Args:
            text: Input text
            language: Language code ('en' or 'nl')

        Returns:
            NERResult with extracted entities
        """
        if language not in self.models:
            logger.warning(f"No model for language '{language}', using English")
            language = "en"

        if language not in self.models:
            return NERResult(
                entities=[],
                entity_counts={},
                text_length=len(text),
                language=language,
                model_used="none"
            )

        model = self.models[language]
        doc = model(text)

        entities = []
        entity_counts = defaultdict(int)

        for ent in doc.ents:
            if ent.label_ not in self.config.entity_types:
                continue

            # Normalize entity text
            normalized = self._normalize_entity(ent.text, ent.label_)

            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence by default
                normalized_text=normalized,
                source_language=language,
            )
            entities.append(entity)
            entity_counts[ent.label_] += 1

        return NERResult(
            entities=entities,
            entity_counts=dict(entity_counts),
            text_length=len(text),
            language=language,
            model_used=model.meta.get("name", "unknown"),
        )

    def extract_batch(
        self,
        texts: List[Tuple[str, str]]
    ) -> List[NERResult]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of (text, language) tuples

        Returns:
            List of NERResults
        """
        results = []
        for text, language in texts:
            result = self.extract(text, language)
            results.append(result)
        return results

    def _normalize_entity(self, text: str, label: str) -> str:
        """
        Normalize entity text for consistency.

        - Remove extra whitespace
        - Standardize formatting
        - Apply label-specific rules
        """
        if not self.config.normalize_entities:
            return text

        # Basic cleanup
        normalized = " ".join(text.split())

        # Label-specific normalization
        if label == "PERSON":
            # Title case for names
            normalized = normalized.title()
        elif label == "ORG":
            # Preserve original case for organizations
            pass
        elif label in ("DATE", "TIME"):
            # Keep as-is for dates
            pass
        elif label == "MONEY":
            # Standardize currency format
            normalized = re.sub(r'\s+', '', normalized)

        if self.config.lowercase_normalize:
            normalized = normalized.lower()

        return normalized


class EntityLinker:
    """
    Link and deduplicate entities across documents.

    Creates a unified entity registry for knowledge graph construction.
    """

    def __init__(self):
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        self.entity_mentions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def register_entity(
        self,
        entity: Entity,
        document_id: str,
        chunk_id: Optional[str] = None
    ) -> str:
        """
        Register an entity and return canonical ID.

        Args:
            entity: Extracted entity
            document_id: Source document ID
            chunk_id: Source chunk ID

        Returns:
            Canonical entity ID
        """
        # Create canonical key
        canonical_key = self._create_canonical_key(entity)

        # Register or update entity
        if canonical_key not in self.entity_registry:
            self.entity_registry[canonical_key] = {
                "entity_id": canonical_key,
                "canonical_name": entity.normalized_text or entity.text,
                "label": entity.label,
                "mention_count": 0,
                "documents": set(),
                "variants": set(),
            }

        # Update registry
        self.entity_registry[canonical_key]["mention_count"] += 1
        self.entity_registry[canonical_key]["documents"].add(document_id)
        self.entity_registry[canonical_key]["variants"].add(entity.text)

        # Record mention
        self.entity_mentions[canonical_key].append({
            "text": entity.text,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        })

        return canonical_key

    def _create_canonical_key(self, entity: Entity) -> str:
        """Create canonical key for entity deduplication."""
        import hashlib

        # Use normalized text and label for key
        text = (entity.normalized_text or entity.text).lower().strip()
        key_string = f"{entity.label}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()[:12]

    def get_entity_graph_nodes(self) -> List[Dict[str, Any]]:
        """Get entities formatted for knowledge graph."""
        nodes = []
        for entity_id, data in self.entity_registry.items():
            nodes.append({
                "id": entity_id,
                "name": data["canonical_name"],
                "type": data["label"],
                "mention_count": data["mention_count"],
                "document_count": len(data["documents"]),
                "variants": list(data["variants"]),
            })
        return nodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get entity extraction statistics."""
        label_counts = defaultdict(int)
        for data in self.entity_registry.values():
            label_counts[data["label"]] += 1

        return {
            "total_unique_entities": len(self.entity_registry),
            "total_mentions": sum(d["mention_count"] for d in self.entity_registry.values()),
            "entities_by_type": dict(label_counts),
            "avg_mentions_per_entity": (
                sum(d["mention_count"] for d in self.entity_registry.values()) /
                len(self.entity_registry) if self.entity_registry else 0
            ),
        }


class DomainGlossaryBuilder:
    """
    Build domain glossary from extracted entities.

    Creates a glossary of domain-specific terms for
    improved retrieval and understanding.
    """

    def __init__(self):
        self.terms: Dict[str, Dict[str, Any]] = {}

    def add_entity(
        self,
        entity: Entity,
        context: str,
        document_id: str
    ):
        """Add entity to glossary with context."""
        term_key = (entity.normalized_text or entity.text).lower()

        if term_key not in self.terms:
            self.terms[term_key] = {
                "term": entity.normalized_text or entity.text,
                "type": entity.label,
                "contexts": [],
                "documents": set(),
                "frequency": 0,
            }

        self.terms[term_key]["frequency"] += 1
        self.terms[term_key]["documents"].add(document_id)

        # Add context snippet (limit to prevent bloat)
        if len(self.terms[term_key]["contexts"]) < 5:
            # Extract surrounding context
            start = max(0, entity.start_char - 100)
            end = min(len(context), entity.end_char + 100)
            context_snippet = context[start:end].strip()
            if context_snippet not in self.terms[term_key]["contexts"]:
                self.terms[term_key]["contexts"].append(context_snippet)

    def get_glossary(self, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Get glossary entries.

        Args:
            min_frequency: Minimum occurrence count

        Returns:
            List of glossary entries
        """
        glossary = []
        for term_key, data in self.terms.items():
            if data["frequency"] >= min_frequency:
                glossary.append({
                    "term": data["term"],
                    "type": data["type"],
                    "frequency": data["frequency"],
                    "document_count": len(data["documents"]),
                    "example_contexts": data["contexts"][:3],
                })

        return sorted(glossary, key=lambda x: x["frequency"], reverse=True)


# Export
__all__ = [
    'MultilingualNER',
    'Entity',
    'NERResult',
    'NERConfig',
    'EntityLinker',
    'DomainGlossaryBuilder',
]
