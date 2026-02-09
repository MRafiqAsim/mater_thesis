"""
GraphRAG Module
===============
Knowledge graph construction and community-based retrieval augmentation.

Components:
- entity_extraction: LLM-based entity and relationship extraction
- graph_store: Cosmos DB Gremlin and in-memory graph storage
- community_detection: Leiden algorithm hierarchical community detection
- community_summarization: GPT-4o community summarization

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from .entity_extraction import (
    GraphRAGEntityExtractor,
    EntityNormalizer,
    RelationshipProcessor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionConfig,
)

from .graph_store import (
    CosmosGraphStore,
    InMemoryGraphStore,
    GraphConfig,
)

from .community_detection import (
    LeidenCommunityDetector,
    CommunityAnalyzer,
    CommunityExporter,
    Community,
    CommunityHierarchy,
    CommunityConfig,
)

from .community_summarization import (
    CommunitySummarizer,
    HierarchicalSummarizer,
    CommunitySummaryIndexer,
    CommunitySummary,
    SummarizationConfig,
)

__all__ = [
    # Entity Extraction
    'GraphRAGEntityExtractor',
    'EntityNormalizer',
    'RelationshipProcessor',
    'ExtractionResult',
    'ExtractedEntity',
    'ExtractedRelationship',
    'ExtractionConfig',
    # Graph Store
    'CosmosGraphStore',
    'InMemoryGraphStore',
    'GraphConfig',
    # Community Detection
    'LeidenCommunityDetector',
    'CommunityAnalyzer',
    'CommunityExporter',
    'Community',
    'CommunityHierarchy',
    'CommunityConfig',
    # Community Summarization
    'CommunitySummarizer',
    'HierarchicalSummarizer',
    'CommunitySummaryIndexer',
    'CommunitySummary',
    'SummarizationConfig',
]
