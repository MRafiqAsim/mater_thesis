"""
GraphRAG Entity Extraction Module
=================================
LLM-based entity and relationship extraction for knowledge graph construction.

Uses GPT-4o with Pydantic structured outputs for reliable extraction.

Entity Types:
- PERSON - People names
- ORGANIZATION - Companies, departments, teams
- PROJECT - Project names and identifiers
- TECHNOLOGY - Technologies, tools, systems
- DOCUMENT - Document references
- EVENT - Meetings, milestones, deadlines
- LOCATION - Places, offices, regions

Relationship Types:
- WORKS_ON - Person works on project
- REPORTS_TO - Reporting relationship
- MENTIONS - Document mentions entity
- USES - Project uses technology
- PART_OF - Entity is part of another
- AUTHORED - Person authored document
- ATTENDED - Person attended event
- LOCATED_IN - Entity located in place

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

from prompt_loader import get_prompt

logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models for Structured Extraction
# ============================================

class ExtractedEntity(BaseModel):
    """Single extracted entity."""
    name: str = Field(description="The entity name as it appears in the text")
    type: str = Field(description="Entity type: PERSON, ORGANIZATION, PROJECT, TECHNOLOGY, DOCUMENT, EVENT, LOCATION")
    description: str = Field(description="Brief description of the entity based on context")


class ExtractedRelationship(BaseModel):
    """Single extracted relationship between entities."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship: str = Field(description="Relationship type: WORKS_ON, REPORTS_TO, MENTIONS, USES, PART_OF, AUTHORED, ATTENDED, LOCATED_IN")
    description: str = Field(description="Brief description of the relationship")
    strength: float = Field(default=1.0, description="Relationship strength (0.0-1.0)")


class ExtractionResult(BaseModel):
    """Complete extraction result from a text chunk."""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


# ============================================
# Entity Extraction Configuration
# ============================================

@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    # Entity types to extract
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORGANIZATION", "PROJECT", "TECHNOLOGY",
        "DOCUMENT", "EVENT", "LOCATION"
    ])

    # Relationship types to extract
    relationship_types: List[str] = field(default_factory=lambda: [
        "WORKS_ON", "REPORTS_TO", "MENTIONS", "USES",
        "PART_OF", "AUTHORED", "ATTENDED", "LOCATED_IN"
    ])

    # Model settings
    model_deployment: str = "gpt-4o"
    temperature: float = 0.0  # Deterministic for consistency
    max_tokens: int = 2000

    # Extraction settings
    max_entities_per_chunk: int = 20
    max_relationships_per_chunk: int = 30
    min_entity_mentions: int = 1

    # Text limits
    max_chunk_length: int = 8000


# ============================================
# Entity Extractor
# ============================================

class GraphRAGEntityExtractor:
    """
    Extract entities and relationships using GPT-4o with structured outputs.

    Usage:
        extractor = GraphRAGEntityExtractor(azure_endpoint, api_key)
        result = extractor.extract("John works on Project Alpha...")
    """

    SYSTEM_PROMPT = get_prompt("gold", "graphrag_entity_extraction", "system_prompt")

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize entity extractor.

        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()

        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ).with_structured_output(ExtractionResult)

        from langchain_core.prompts import ChatPromptTemplate

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", get_prompt("gold", "graphrag_entity_extraction", "user_prompt")),
        ])

        self.chain = self.prompt | self.llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ExtractionResult, Dict[str, Any]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from
            chunk_id: Source chunk identifier
            metadata: Additional metadata

        Returns:
            Tuple of (ExtractionResult, extraction_metadata)
        """
        # Truncate if too long
        if len(text) > self.config.max_chunk_length:
            text = text[:self.config.max_chunk_length]

        # Extract
        result = self.chain.invoke({
            "text": text,
            "max_entities": self.config.max_entities_per_chunk,
            "max_relationships": self.config.max_relationships_per_chunk,
        })

        # Build metadata
        extraction_meta = {
            "chunk_id": chunk_id,
            "num_entities": len(result.entities),
            "num_relationships": len(result.relationships),
            "text_length": len(text),
            "source_metadata": metadata or {},
        }

        return result, extraction_meta

    def extract_batch(
        self,
        texts: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[ExtractionResult, Dict[str, Any]]]:
        """
        Extract from multiple texts.

        Args:
            texts: List of {"text": ..., "chunk_id": ..., "metadata": ...}
            progress_callback: Optional callback(current, total)

        Returns:
            List of (ExtractionResult, metadata) tuples
        """
        results = []
        total = len(texts)

        for i, item in enumerate(texts):
            try:
                result, meta = self.extract(
                    text=item.get("text", ""),
                    chunk_id=item.get("chunk_id"),
                    metadata=item.get("metadata"),
                )
                results.append((result, meta))
            except Exception as e:
                logger.warning(f"Extraction failed for {item.get('chunk_id')}: {e}")
                # Return empty result on failure
                results.append((
                    ExtractionResult(entities=[], relationships=[]),
                    {"chunk_id": item.get("chunk_id"), "error": str(e)}
                ))

            if progress_callback:
                progress_callback(i + 1, total)

        return results


# ============================================
# Entity Normalization & Deduplication
# ============================================

class EntityNormalizer:
    """
    Normalize and deduplicate extracted entities.

    Creates canonical entity IDs and merges duplicates.
    """

    def __init__(self):
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        self.name_to_id: Dict[str, str] = {}

    def normalize_name(self, name: str, entity_type: str) -> str:
        """Normalize entity name."""
        # Basic normalization
        normalized = " ".join(name.split()).strip()

        # Type-specific normalization
        if entity_type == "PERSON":
            normalized = normalized.title()
        elif entity_type == "ORGANIZATION":
            # Keep original case for organizations
            pass
        elif entity_type == "TECHNOLOGY":
            # Keep original case for technologies
            pass

        return normalized

    def get_entity_id(self, name: str, entity_type: str) -> str:
        """Get or create canonical entity ID."""
        normalized = self.normalize_name(name, entity_type)
        key = f"{entity_type}:{normalized.lower()}"

        if key in self.name_to_id:
            return self.name_to_id[key]

        # Create new ID
        entity_id = hashlib.md5(key.encode()).hexdigest()[:12]
        self.name_to_id[key] = entity_id

        return entity_id

    def register_entity(
        self,
        entity: ExtractedEntity,
        chunk_id: str,
        source_file: Optional[str] = None
    ) -> str:
        """
        Register an entity and return its canonical ID.

        Args:
            entity: Extracted entity
            chunk_id: Source chunk ID
            source_file: Source file name

        Returns:
            Canonical entity ID
        """
        entity_id = self.get_entity_id(entity.name, entity.type)

        if entity_id not in self.entity_registry:
            self.entity_registry[entity_id] = {
                "id": entity_id,
                "name": self.normalize_name(entity.name, entity.type),
                "type": entity.type,
                "description": entity.description,
                "mention_count": 0,
                "chunks": set(),
                "sources": set(),
                "variants": set(),
            }

        # Update registry
        self.entity_registry[entity_id]["mention_count"] += 1
        self.entity_registry[entity_id]["chunks"].add(chunk_id)
        self.entity_registry[entity_id]["variants"].add(entity.name)

        if source_file:
            self.entity_registry[entity_id]["sources"].add(source_file)

        # Update description if longer
        if len(entity.description) > len(self.entity_registry[entity_id]["description"]):
            self.entity_registry[entity_id]["description"] = entity.description

        return entity_id

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all registered entities."""
        entities = []
        for entity_id, data in self.entity_registry.items():
            entities.append({
                "id": data["id"],
                "name": data["name"],
                "type": data["type"],
                "description": data["description"],
                "mention_count": data["mention_count"],
                "chunk_count": len(data["chunks"]),
                "source_count": len(data["sources"]),
                "variants": list(data["variants"]),
            })
        return entities

    def get_statistics(self) -> Dict[str, Any]:
        """Get entity statistics."""
        from collections import Counter

        type_counts = Counter(e["type"] for e in self.entity_registry.values())
        mention_counts = [e["mention_count"] for e in self.entity_registry.values()]

        return {
            "total_entities": len(self.entity_registry),
            "by_type": dict(type_counts),
            "avg_mentions": sum(mention_counts) / len(mention_counts) if mention_counts else 0,
            "max_mentions": max(mention_counts) if mention_counts else 0,
        }


# ============================================
# Relationship Processing
# ============================================

class RelationshipProcessor:
    """
    Process and deduplicate extracted relationships.
    """

    def __init__(self, entity_normalizer: EntityNormalizer):
        self.normalizer = entity_normalizer
        self.relationships: List[Dict[str, Any]] = []
        self.relationship_set: set = set()  # For deduplication

    def add_relationship(
        self,
        relationship: ExtractedRelationship,
        chunk_id: str,
        source_entities: Dict[str, str],  # name -> entity_id mapping
        target_entities: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Add a relationship with entity resolution.

        Returns relationship dict if added, None if duplicate.
        """
        # Resolve source entity
        source_id = source_entities.get(relationship.source)
        if not source_id:
            # Try to find by normalized name
            source_id = self.normalizer.get_entity_id(relationship.source, "UNKNOWN")

        # Resolve target entity
        target_id = target_entities.get(relationship.target)
        if not target_id:
            target_id = self.normalizer.get_entity_id(relationship.target, "UNKNOWN")

        # Create relationship key for deduplication
        rel_key = f"{source_id}:{relationship.relationship}:{target_id}"

        if rel_key in self.relationship_set:
            return None

        self.relationship_set.add(rel_key)

        rel_dict = {
            "id": hashlib.md5(rel_key.encode()).hexdigest()[:12],
            "source_id": source_id,
            "source_name": relationship.source,
            "target_id": target_id,
            "target_name": relationship.target,
            "type": relationship.relationship,
            "description": relationship.description,
            "strength": relationship.strength,
            "chunk_id": chunk_id,
        }

        self.relationships.append(rel_dict)
        return rel_dict

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all processed relationships."""
        return self.relationships

    def get_statistics(self) -> Dict[str, Any]:
        """Get relationship statistics."""
        from collections import Counter

        type_counts = Counter(r["type"] for r in self.relationships)

        return {
            "total_relationships": len(self.relationships),
            "unique_relationships": len(self.relationship_set),
            "by_type": dict(type_counts),
        }


# Export
__all__ = [
    'GraphRAGEntityExtractor',
    'EntityNormalizer',
    'RelationshipProcessor',
    'ExtractionResult',
    'ExtractedEntity',
    'ExtractedRelationship',
    'ExtractionConfig',
]
