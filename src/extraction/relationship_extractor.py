"""
Knowledge Graph Relationship Extractor Module

Extracts relationships between entities for PathRAG knowledge graph.
Supports multiple strategies for benchmarking:
- CooccurrenceExtractor: Fast, rule-based (entities in same sentence)
- LLMRelationshipExtractor: LLM-based (more accurate, extracts semantics)
- HybridRelationshipExtractor: Combines both approaches

PathRAG Relationship Format:
    (source_entity, target_entity, description, keywords, weight)

Usage:
    extractor = LLMRelationshipExtractor(api_key="...")
    relationships = extractor.extract(text, entities, language="en")
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from .kg_entity_extractor import KGEntity

logger = logging.getLogger(__name__)


@dataclass
class KGRelationship:
    """A relationship between two entities for PathRAG knowledge graph"""
    source: str           # Source entity text
    target: str           # Target entity text
    source_type: str      # Source entity type (PathRAG type)
    target_type: str      # Target entity type (PathRAG type)
    description: str      # Relationship description
    keywords: List[str]   # High-level keywords categorizing the relationship
    weight: float = 1.0   # Relationship strength (0-1)
    source_method: str = "cooccurrence"  # Extraction method for benchmarking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "description": self.description,
            "keywords": self.keywords,
            "weight": self.weight,
            "extraction_method": self.source_method,
        }

    def to_pathrag_format(self) -> Dict[str, Any]:
        """Convert to PathRAG's expected format"""
        return {
            "src_id": self.source,
            "tgt_id": self.target,
            "description": self.description,
            "keywords": ",".join(self.keywords),
            "weight": self.weight,
        }


class RelationshipExtractor(ABC):
    """
    Abstract base class for relationship extractors.

    Implement this interface to create new extraction strategies
    for benchmarking different approaches.
    """

    @abstractmethod
    def extract(
        self,
        text: str,
        entities: List[KGEntity],
        language: str = "en"
    ) -> List[KGRelationship]:
        """
        Extract relationships between entities in text.

        Args:
            text: Source text
            entities: List of entities already extracted from text
            language: Language code

        Returns:
            List of KGRelationship objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for logging/benchmarking"""
        pass


class CooccurrenceRelationshipExtractor(RelationshipExtractor):
    """
    Rule-based relationship extractor using co-occurrence.

    Fast extraction based on entities appearing in the same sentence.
    Good for initial relationship discovery, but lacks semantic understanding.
    """

    def __init__(
        self,
        window_size: int = 1,  # Number of sentences to consider
        min_weight: float = 0.3
    ):
        """
        Initialize co-occurrence extractor.

        Args:
            window_size: Number of sentences for co-occurrence window
            min_weight: Minimum weight threshold for relationships
        """
        self.window_size = window_size
        self.min_weight = min_weight

    def extract(
        self,
        text: str,
        entities: List[KGEntity],
        language: str = "en"
    ) -> List[KGRelationship]:
        """Extract relationships based on co-occurrence in sentences"""
        relationships = []

        if len(entities) < 2:
            return relationships

        # Split text into sentences
        sentences = self._split_sentences(text)

        # Find which entities appear in which sentences
        entity_sentences: Dict[str, Set[int]] = {}
        for entity in entities:
            entity_key = entity.text.lower()
            entity_sentences[entity_key] = set()

            for i, sentence in enumerate(sentences):
                if entity.text.lower() in sentence.lower():
                    # Add this sentence and neighbors within window
                    for j in range(max(0, i - self.window_size),
                                   min(len(sentences), i + self.window_size + 1)):
                        entity_sentences[entity_key].add(j)

        # Find co-occurring entity pairs
        seen_pairs: Set[Tuple[str, str]] = set()
        entity_map = {e.text.lower(): e for e in entities}

        for e1 in entities:
            for e2 in entities:
                if e1.text == e2.text:
                    continue

                # Normalize pair order to avoid duplicates
                pair = tuple(sorted([e1.text.lower(), e2.text.lower()]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Check co-occurrence
                e1_sentences = entity_sentences.get(e1.text.lower(), set())
                e2_sentences = entity_sentences.get(e2.text.lower(), set())
                overlap = e1_sentences & e2_sentences

                if overlap:
                    # Calculate weight based on co-occurrence frequency
                    weight = min(1.0, len(overlap) * 0.3)

                    if weight >= self.min_weight:
                        # Generate simple description
                        description = f"{e1.text} and {e2.text} appear together in context"

                        # Generate keywords based on entity types
                        keywords = self._generate_keywords(e1, e2)

                        relationships.append(KGRelationship(
                            source=e1.text,
                            target=e2.text,
                            source_type=e1.pathrag_type or e1.entity_type.lower(),
                            target_type=e2.pathrag_type or e2.entity_type.lower(),
                            description=description,
                            keywords=keywords,
                            weight=weight,
                            source_method="cooccurrence"
                        ))

        return relationships

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_keywords(self, e1: KGEntity, e2: KGEntity) -> List[str]:
        """Generate relationship keywords based on entity types"""
        keywords = []

        type_pair = {e1.pathrag_type or e1.entity_type.lower(),
                     e2.pathrag_type or e2.entity_type.lower()}

        if "person" in type_pair and "organization" in type_pair:
            keywords.extend(["employment", "affiliation"])
        elif "person" in type_pair and "geo" in type_pair:
            keywords.extend(["location", "based_in"])
        elif "organization" in type_pair and "geo" in type_pair:
            keywords.extend(["headquarters", "operates_in"])
        elif "person" in type_pair and "event" in type_pair:
            keywords.extend(["participation", "involvement"])
        elif "organization" in type_pair and "product" in type_pair:
            keywords.extend(["produces", "develops"])
        else:
            keywords.append("related_to")

        return keywords

    @property
    def name(self) -> str:
        return "CooccurrenceRelationshipExtractor"


class LLMRelationshipExtractor(RelationshipExtractor):
    """
    LLM-based relationship extractor.

    Uses OpenAI/Azure OpenAI for semantic relationship extraction.
    More accurate but slower and requires API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_entities_per_call: int = 20,
        # Azure OpenAI settings
        use_azure: bool = False,
        azure_endpoint: str = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str = None
    ):
        """
        Initialize LLM relationship extractor.

        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model to use (for OpenAI) or deployment name (for Azure)
            max_entities_per_call: Maximum entities to process per API call
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name (defaults to model)
        """
        self.api_key = api_key
        self.model = model
        self.max_entities_per_call = max_entities_per_call
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

    def extract(
        self,
        text: str,
        entities: List[KGEntity],
        language: str = "en"
    ) -> List[KGRelationship]:
        """Extract relationships using LLM"""
        if not self.api_key or len(entities) < 2:
            return []

        relationships = []

        try:
            import json
            client = self._get_client()

            # Prepare entity list for prompt
            entity_list = "\n".join([
                f"- {e.text} ({e.pathrag_type or e.entity_type})"
                for e in entities[:self.max_entities_per_call]
            ])

            prompt = f"""Analyze this text and extract relationships between the listed entities.

TEXT:
{text[:3000]}

ENTITIES:
{entity_list}

For each relationship found, provide:
1. source: Source entity name (exactly as listed)
2. target: Target entity name (exactly as listed)
3. description: Brief description of the relationship
4. keywords: 1-3 keywords categorizing the relationship type
5. weight: Strength of relationship (0.0-1.0)

Return a JSON array of relationship objects. Only include clear, meaningful relationships.
Example format:
[
  {{"source": "John Smith", "target": "Microsoft", "description": "works at", "keywords": ["employment"], "weight": 0.9}},
  {{"source": "Microsoft", "target": "Seattle", "description": "headquartered in", "keywords": ["location", "headquarters"], "weight": 0.8}}
]

Return ONLY the JSON array, no other text. If no relationships found, return empty array []."""

            # Use deployment name for Azure, model name for OpenAI
            model_name = self.azure_deployment if self.use_azure else self.model
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks (```json ... ```)
            if content.startswith("```"):
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group()

            # Parse JSON response
            if content.startswith("["):
                raw_relationships = json.loads(content)

                # Create entity lookup for type information
                entity_map = {e.text.lower(): e for e in entities}

                for rel in raw_relationships:
                    source_entity = entity_map.get(rel.get("source", "").lower())
                    target_entity = entity_map.get(rel.get("target", "").lower())

                    if source_entity and target_entity:
                        keywords = rel.get("keywords", [])
                        if isinstance(keywords, str):
                            keywords = [keywords]

                        relationships.append(KGRelationship(
                            source=rel.get("source", ""),
                            target=rel.get("target", ""),
                            source_type=source_entity.pathrag_type or source_entity.entity_type.lower(),
                            target_type=target_entity.pathrag_type or target_entity.entity_type.lower(),
                            description=rel.get("description", ""),
                            keywords=keywords,
                            weight=float(rel.get("weight", 0.5)),
                            source_method="llm"
                        ))

        except Exception as e:
            logger.error(f"LLM relationship extraction error: {e}")

        return relationships

    @property
    def name(self) -> str:
        return f"LLMRelationshipExtractor({self.model})"


class HybridRelationshipExtractor(RelationshipExtractor):
    """
    Hybrid relationship extractor combining co-occurrence and LLM.

    Uses co-occurrence for fast initial extraction, then LLM to
    enhance descriptions and verify relationships.
    """

    def __init__(
        self,
        cooccurrence_extractor: CooccurrenceRelationshipExtractor,
        llm_extractor: Optional[LLMRelationshipExtractor] = None,
        use_llm_for_descriptions: bool = True
    ):
        """
        Initialize hybrid extractor.

        Args:
            cooccurrence_extractor: Primary fast extractor
            llm_extractor: Optional LLM extractor for enhancement
            use_llm_for_descriptions: Use LLM to improve descriptions
        """
        self.cooccurrence_extractor = cooccurrence_extractor
        self.llm_extractor = llm_extractor
        self.use_llm_for_descriptions = use_llm_for_descriptions

    def extract(
        self,
        text: str,
        entities: List[KGEntity],
        language: str = "en"
    ) -> List[KGRelationship]:
        """Extract using co-occurrence, optionally enhance with LLM"""
        # Primary extraction with co-occurrence
        relationships = self.cooccurrence_extractor.extract(text, entities, language)

        # Optional LLM enhancement
        if self.llm_extractor and self.use_llm_for_descriptions:
            llm_relationships = self.llm_extractor.extract(text, entities, language)

            # Merge LLM relationships, preferring LLM descriptions
            cooc_map = {(r.source.lower(), r.target.lower()): r for r in relationships}

            for llm_rel in llm_relationships:
                key = (llm_rel.source.lower(), llm_rel.target.lower())
                reverse_key = (llm_rel.target.lower(), llm_rel.source.lower())

                if key in cooc_map:
                    # Update existing relationship with LLM description
                    existing = cooc_map[key]
                    existing.description = llm_rel.description
                    existing.keywords = llm_rel.keywords
                    existing.weight = max(existing.weight, llm_rel.weight)
                    existing.source_method = "hybrid"
                elif reverse_key in cooc_map:
                    existing = cooc_map[reverse_key]
                    existing.description = llm_rel.description
                    existing.keywords = llm_rel.keywords
                    existing.weight = max(existing.weight, llm_rel.weight)
                    existing.source_method = "hybrid"
                else:
                    # New relationship from LLM
                    llm_rel.source_method = "llm"
                    relationships.append(llm_rel)

        return relationships

    @property
    def name(self) -> str:
        return "HybridRelationshipExtractor"


# Factory function for easy creation
def create_relationship_extractor(
    strategy: str = "cooccurrence",
    openai_api_key: str = None,
    # Azure OpenAI settings
    use_azure: bool = False,
    azure_endpoint: str = None,
    azure_api_version: str = "2024-12-01-preview",
    azure_deployment: str = None,
    **kwargs
) -> RelationshipExtractor:
    """
    Factory function to create relationship extractor.

    Args:
        strategy: "cooccurrence", "llm", or "hybrid"
        openai_api_key: Required for LLM strategies
        use_azure: Whether to use Azure OpenAI
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_version: Azure API version
        azure_deployment: Azure deployment name

    Returns:
        Configured RelationshipExtractor instance
    """
    if strategy == "cooccurrence":
        return CooccurrenceRelationshipExtractor(**kwargs)

    elif strategy == "llm":
        if not openai_api_key:
            raise ValueError("openai_api_key required for LLM strategy")
        return LLMRelationshipExtractor(
            api_key=openai_api_key,
            use_azure=use_azure,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            **kwargs
        )

    elif strategy == "hybrid":
        cooc_ext = CooccurrenceRelationshipExtractor()
        llm_ext = None
        if openai_api_key:
            llm_ext = LLMRelationshipExtractor(
                api_key=openai_api_key,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment
            )
        return HybridRelationshipExtractor(
            cooccurrence_extractor=cooc_ext,
            llm_extractor=llm_ext,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
