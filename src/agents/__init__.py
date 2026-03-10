"""
Agents Module
=============
ReAct agent and tools for tool-augmented reasoning.

Components:
- graphrag_retriever: Combined GraphRAG + Vector retrieval
- tools: LangChain tools for agent actions
- react_agent: ReAct reasoning agent with LangGraph

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from .graphrag_retriever import (
    GraphRAGRetriever,
    HybridRetriever,
    GraphRAGContext,
    GraphRAGConfig,
    RetrievalResult,
    QueryType,
    QueryClassifier,
)

from .tools import (
    VectorSearchTool,
    EntityLookupTool,
    RelationshipSearchTool,
    CommunitySummaryTool,
    GraphTraversalTool,
    create_agent_tools,
)

__all__ = [
    # GraphRAG Retriever
    'GraphRAGRetriever',
    'HybridRetriever',
    'GraphRAGContext',
    'GraphRAGConfig',
    'RetrievalResult',
    'QueryType',
    'QueryClassifier',
    # Tools
    'VectorSearchTool',
    'EntityLookupTool',
    'RelationshipSearchTool',
    'CommunitySummaryTool',
    'GraphTraversalTool',
    'create_agent_tools',
]
