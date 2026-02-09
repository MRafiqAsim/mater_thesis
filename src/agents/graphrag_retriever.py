"""
GraphRAG Retriever Module
=========================
Combined retrieval integrating vector search, graph traversal, and community summaries.

Features:
- Hybrid vector + keyword search
- Knowledge graph entity lookup
- Community summary retrieval
- Multi-source context aggregation
- Query routing (local vs global)

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification for routing."""
    LOCAL = "local"      # Specific entity/fact queries
    GLOBAL = "global"    # Broad theme/summary queries
    HYBRID = "hybrid"    # Mixed queries


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    content: str
    source_type: str  # "chunk", "entity", "community", "relationship"
    source_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRAGContext:
    """Aggregated context from multiple retrieval sources."""
    chunks: List[RetrievalResult]
    entities: List[RetrievalResult]
    communities: List[RetrievalResult]
    relationships: List[RetrievalResult]
    query_type: QueryType
    total_tokens: int = 0

    def get_all_results(self) -> List[RetrievalResult]:
        """Get all results sorted by score."""
        all_results = self.chunks + self.entities + self.communities + self.relationships
        return sorted(all_results, key=lambda x: x.score, reverse=True)

    def format_context(self, max_tokens: int = 8000) -> str:
        """Format context for LLM prompt."""
        sections = []

        # Add community summaries (high-level context)
        if self.communities:
            community_text = "\n\n".join([
                f"[Community {r.source_id}]: {r.content}"
                for r in self.communities[:3]
            ])
            sections.append(f"## Relevant Topics\n{community_text}")

        # Add entity information
        if self.entities:
            entity_text = "\n".join([
                f"- {r.content}"
                for r in self.entities[:10]
            ])
            sections.append(f"## Key Entities\n{entity_text}")

        # Add relationships
        if self.relationships:
            rel_text = "\n".join([
                f"- {r.content}"
                for r in self.relationships[:10]
            ])
            sections.append(f"## Relationships\n{rel_text}")

        # Add document chunks (detailed evidence)
        if self.chunks:
            chunk_text = "\n\n---\n\n".join([
                f"[Source: {r.metadata.get('source_file', r.source_id)}]\n{r.content}"
                for r in self.chunks[:5]
            ])
            sections.append(f"## Source Documents\n{chunk_text}")

        return "\n\n".join(sections)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG retriever."""
    # Vector search settings
    vector_top_k: int = 10
    vector_score_threshold: float = 0.7

    # Graph search settings
    entity_top_k: int = 10
    max_graph_hops: int = 2
    relationship_limit: int = 20

    # Community search settings
    community_top_k: int = 5
    prefer_level: int = 1  # Default community level

    # Query routing
    local_keywords: List[str] = field(default_factory=lambda: [
        "who", "what", "when", "where", "which", "name", "specific"
    ])
    global_keywords: List[str] = field(default_factory=lambda: [
        "overview", "summary", "general", "trends", "themes", "overall", "main"
    ])

    # Context limits
    max_context_tokens: int = 8000


class QueryClassifier:
    """Classify queries for routing."""

    def __init__(self, config: GraphRAGConfig):
        self.config = config

    def classify(self, query: str) -> QueryType:
        """
        Classify query as local, global, or hybrid.

        Local: Specific entity/fact questions
        Global: Theme/summary questions
        Hybrid: Mixed or unclear
        """
        query_lower = query.lower()

        local_score = sum(1 for kw in self.config.local_keywords if kw in query_lower)
        global_score = sum(1 for kw in self.config.global_keywords if kw in query_lower)

        # Check for entity mentions (capitalized words)
        words = query.split()
        entity_mentions = sum(1 for w in words if w[0].isupper() and len(w) > 2)
        local_score += entity_mentions

        # Question length heuristic (longer = more global)
        if len(words) > 15:
            global_score += 1

        if local_score > global_score + 1:
            return QueryType.LOCAL
        elif global_score > local_score + 1:
            return QueryType.GLOBAL
        else:
            return QueryType.HYBRID


class GraphRAGRetriever:
    """
    Combined retriever integrating vector search, graph, and community summaries.

    Usage:
        retriever = GraphRAGRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            community_store=community_store,
            embeddings=embeddings
        )
        context = retriever.retrieve("What projects is John working on?")
    """

    def __init__(
        self,
        vector_store,  # Azure AI Search or similar
        graph_store,   # CosmosGraphStore or InMemoryGraphStore
        community_summaries: List[Dict[str, Any]],
        community_embeddings: List[List[float]],
        embeddings,    # Embedding model
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        config: Optional[GraphRAGConfig] = None
    ):
        """
        Initialize GraphRAG retriever.

        Args:
            vector_store: Vector store for chunk retrieval
            graph_store: Graph store for entity/relationship queries
            community_summaries: List of community summary dicts
            community_embeddings: Pre-computed community embeddings
            embeddings: Embedding model for queries
            entities: List of entity dicts
            relationships: List of relationship dicts
            config: Retriever configuration
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.community_summaries = community_summaries
        self.community_embeddings = np.array(community_embeddings)
        self.embeddings = embeddings
        self.entities = {e["id"]: e for e in entities}
        self.entity_names = {e["name"].lower(): e["id"] for e in entities}
        self.relationships = relationships
        self.config = config or GraphRAGConfig()
        self.classifier = QueryClassifier(self.config)

    def retrieve(self, query: str) -> GraphRAGContext:
        """
        Retrieve relevant context for a query.

        Combines vector search, graph traversal, and community summaries
        based on query type.

        Args:
            query: User query

        Returns:
            GraphRAGContext with aggregated results
        """
        # Classify query
        query_type = self.classifier.classify(query)
        logger.info(f"Query classified as: {query_type.value}")

        # Initialize results
        chunks = []
        entities = []
        communities = []
        relationships = []

        # Route based on query type
        if query_type == QueryType.LOCAL:
            # Focus on specific entities and chunks
            chunks = self._vector_search(query)
            entities = self._entity_search(query)
            relationships = self._relationship_search(query, entities)
            communities = self._community_search(query, top_k=2)

        elif query_type == QueryType.GLOBAL:
            # Focus on community summaries and themes
            communities = self._community_search(query, top_k=self.config.community_top_k)
            chunks = self._vector_search(query, top_k=3)
            entities = self._entity_search(query, top_k=5)

        else:  # HYBRID
            # Balanced retrieval
            chunks = self._vector_search(query)
            entities = self._entity_search(query)
            communities = self._community_search(query)
            relationships = self._relationship_search(query, entities)

        return GraphRAGContext(
            chunks=chunks,
            entities=entities,
            communities=communities,
            relationships=relationships,
            query_type=query_type
        )

    def _vector_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Perform vector similarity search on chunks."""
        top_k = top_k or self.config.vector_top_k

        try:
            # Use vector store's search method
            results = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )

            return [
                RetrievalResult(
                    content=doc.page_content,
                    source_type="chunk",
                    source_id=doc.metadata.get("chunk_id", "unknown"),
                    score=score,
                    metadata=doc.metadata
                )
                for doc, score in results
                if score >= self.config.vector_score_threshold
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _entity_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search for relevant entities."""
        top_k = top_k or self.config.entity_top_k
        results = []

        # Extract potential entity mentions from query
        query_lower = query.lower()
        words = query.split()

        # Direct name matching
        for word in words:
            word_lower = word.lower().strip(".,?!")
            if word_lower in self.entity_names:
                entity_id = self.entity_names[word_lower]
                entity = self.entities[entity_id]
                results.append(RetrievalResult(
                    content=f"{entity['name']} ({entity['type']}): {entity.get('description', '')}",
                    source_type="entity",
                    source_id=entity_id,
                    score=1.0,
                    metadata=entity
                ))

        # Fuzzy matching for longer names
        for name, entity_id in self.entity_names.items():
            if len(name) > 3 and name in query_lower:
                entity = self.entities[entity_id]
                if entity_id not in [r.source_id for r in results]:
                    results.append(RetrievalResult(
                        content=f"{entity['name']} ({entity['type']}): {entity.get('description', '')}",
                        source_type="entity",
                        source_id=entity_id,
                        score=0.9,
                        metadata=entity
                    ))

        return results[:top_k]

    def _relationship_search(
        self,
        query: str,
        entity_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Find relationships involving retrieved entities."""
        results = []
        entity_ids = {r.source_id for r in entity_results}

        for rel in self.relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")

            if source_id in entity_ids or target_id in entity_ids:
                source_name = rel.get("source_name", source_id)
                target_name = rel.get("target_name", target_id)
                rel_type = rel.get("type", "RELATED_TO")
                description = rel.get("description", "")

                results.append(RetrievalResult(
                    content=f"{source_name} --[{rel_type}]--> {target_name}: {description}",
                    source_type="relationship",
                    source_id=rel.get("id", f"{source_id}_{target_id}"),
                    score=rel.get("strength", 0.8),
                    metadata=rel
                ))

        # Sort by strength and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.config.relationship_limit]

    def _community_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search community summaries by semantic similarity."""
        top_k = top_k or self.config.community_top_k

        try:
            # Generate query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))

            # Cosine similarity with all community embeddings
            similarities = np.dot(self.community_embeddings, query_embedding) / (
                np.linalg.norm(self.community_embeddings, axis=1) *
                np.linalg.norm(query_embedding)
            )

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                community = self.community_summaries[idx]
                results.append(RetrievalResult(
                    content=community.get("summary", ""),
                    source_type="community",
                    source_id=community.get("community_id", f"comm_{idx}"),
                    score=float(similarities[idx]),
                    metadata={
                        "level": community.get("level", 0),
                        "key_themes": community.get("key_themes", []),
                        "key_entities": community.get("key_entities", []),
                        "member_count": community.get("member_count", 0)
                    }
                ))

            return results

        except Exception as e:
            logger.warning(f"Community search failed: {e}")
            return []

    def retrieve_for_entity(
        self,
        entity_id: str,
        include_neighbors: bool = True
    ) -> GraphRAGContext:
        """
        Retrieve context focused on a specific entity.

        Args:
            entity_id: Entity ID to focus on
            include_neighbors: Whether to include neighbor entities

        Returns:
            GraphRAGContext focused on the entity
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return GraphRAGContext(
                chunks=[], entities=[], communities=[], relationships=[],
                query_type=QueryType.LOCAL
            )

        # Get entity as result
        entity_results = [RetrievalResult(
            content=f"{entity['name']} ({entity['type']}): {entity.get('description', '')}",
            source_type="entity",
            source_id=entity_id,
            score=1.0,
            metadata=entity
        )]

        # Get relationships
        relationship_results = self._relationship_search(entity['name'], entity_results)

        # Get neighbor entities if requested
        if include_neighbors and hasattr(self.graph_store, 'get_neighbors'):
            neighbors = self.graph_store.get_neighbors(entity_id, max_hops=1)
            for neighbor in neighbors[:10]:
                neighbor_id = neighbor.get("id")
                if neighbor_id and neighbor_id in self.entities:
                    n = self.entities[neighbor_id]
                    entity_results.append(RetrievalResult(
                        content=f"{n['name']} ({n['type']}): {n.get('description', '')}",
                        source_type="entity",
                        source_id=neighbor_id,
                        score=0.8,
                        metadata=n
                    ))

        # Search chunks mentioning the entity
        chunk_results = self._vector_search(entity['name'], top_k=5)

        # Find relevant communities
        community_results = self._community_search(entity['name'], top_k=2)

        return GraphRAGContext(
            chunks=chunk_results,
            entities=entity_results,
            communities=community_results,
            relationships=relationship_results,
            query_type=QueryType.LOCAL
        )


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.

    Implements RRF (Reciprocal Rank Fusion) for combining results.
    """

    def __init__(
        self,
        dense_retriever,   # Vector-based
        sparse_retriever,  # Keyword-based (BM25)
        rrf_k: int = 60
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve using RRF fusion of dense and sparse results.
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)

        # Build RRF scores
        rrf_scores = {}

        for rank, result in enumerate(dense_results):
            doc_id = result.source_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)

        for rank, result in enumerate(sparse_results):
            doc_id = result.source_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)

        # Combine and sort
        all_results = {r.source_id: r for r in dense_results + sparse_results}
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [all_results[doc_id] for doc_id in sorted_ids[:top_k] if doc_id in all_results]


# Export
__all__ = [
    'GraphRAGRetriever',
    'HybridRetriever',
    'GraphRAGContext',
    'GraphRAGConfig',
    'RetrievalResult',
    'QueryType',
    'QueryClassifier',
]
