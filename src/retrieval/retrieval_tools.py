"""
Retrieval Tools Module

Defines tools available to the ReAct agent for knowledge retrieval.
Each tool wraps a specific retrieval strategy (PathRAG, GraphRAG, Vector, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    data: Any
    message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "execution_time": self.execution_time
        }


@dataclass
class Tool:
    """Definition of a tool available to the ReAct agent."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, description, required}
    function: Callable

    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function schema."""
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", "")
            }
            if param_info.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class RetrievalToolkit:
    """
    Collection of retrieval tools for the ReAct agent.

    Provides unified access to:
    - PathRAG: Path-based reasoning retrieval
    - GraphRAG: Community-based retrieval
    - Vector Search: Similarity-based retrieval
    - Entity/Chunk lookup
    - Temporal filtering
    """

    def __init__(
        self,
        gold_path: str,
        silver_path: Optional[str] = None,
        mode: str = "llm",
    ):
        """
        Initialize the retrieval toolkit.

        Args:
            gold_path: Path to Gold layer with graph and indexes
            silver_path: Path to Silver layer with chunks (optional)
            mode: Processing mode — "local" uses local models
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None
        self.mode = mode

        # Lazy-loaded components
        self._graph = None
        self._path_indexer = None
        self._community_detector = None
        self._embedding_generator = None
        self._chunk_embeddings = None
        self._chunk_ids = None
        self._entity_embeddings = None
        self._entity_ids = None

        # Register tools
        self.tools: Dict[str, Tool] = {}
        self._register_tools()

        logger.info("RetrievalToolkit initialized")

    def _register_tools(self):
        """Register all available tools."""

        # PathRAG Search
        self.tools["pathrag_search"] = Tool(
            name="pathrag_search",
            description="Find reasoning paths between entities in the knowledge graph. "
                       "Use this for multi-hop queries that connect different concepts.",
            parameters={
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity names to find paths between",
                    "required": True
                },
                "max_paths": {
                    "type": "integer",
                    "description": "Maximum number of paths to return (default: 5)",
                    "required": False
                }
            },
            function=self.pathrag_search
        )

        # GraphRAG Search
        self.tools["graphrag_search"] = Tool(
            name="graphrag_search",
            description="Search for relevant community summaries and their source chunks. "
                       "Use this for broad topic queries or when you need context about a domain.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True
                },
                "level": {
                    "type": "integer",
                    "description": "Community hierarchy level (0=fine, higher=coarse). Default: 0",
                    "required": False
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of communities to return (default: 3)",
                    "required": False
                }
            },
            function=self.graphrag_search
        )

        # Vector Search
        self.tools["vector_search"] = Tool(
            name="vector_search",
            description="Find text chunks most similar to a query using semantic similarity. "
                       "Use this for finding specific information or evidence.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default: 10)",
                    "required": False
                },
                "filter_thread": {
                    "type": "string",
                    "description": "Filter to specific thread ID (optional)",
                    "required": False
                }
            },
            function=self.vector_search
        )

        # Entity Lookup
        self.tools["entity_lookup"] = Tool(
            name="entity_lookup",
            description="Get details about a specific entity including its relationships. "
                       "Use this to understand an entity's role in the knowledge graph.",
            parameters={
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to look up",
                    "required": True
                }
            },
            function=self.entity_lookup
        )

        # List Entities by Type
        self.tools["list_entities"] = Tool(
            name="list_entities",
            description="List all entities of a given type from the knowledge graph. "
                       "Use this for aggregate questions like 'list all projects', 'who are the people', "
                       "'what organizations are mentioned'. "
                       "Valid types: PERSON, ORG, PRODUCT, GPE, DOCUMENT, EVENT, LOC, FAC, WORK_OF_ART.",
            parameters={
                "entity_type": {
                    "type": "string",
                    "description": "The entity type to list (e.g., PRODUCT, PERSON, ORG)",
                    "required": True
                }
            },
            function=self.list_entities
        )

        # List Entity Types
        self.tools["list_entity_types"] = Tool(
            name="list_entity_types",
            description="Discover what entity types exist in the knowledge graph and how many of each. "
                       "Use this FIRST for aggregate/listing queries to understand what data is available "
                       "before searching for specific types.",
            parameters={},
            function=self.list_entity_types
        )

        # Get Chunk Context
        self.tools["get_chunk_context"] = Tool(
            name="get_chunk_context",
            description="Get the full context around a chunk including thread and attachments. "
                       "Use this when you need more context about a specific piece of evidence.",
            parameters={
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk ID to get context for",
                    "required": True
                }
            },
            function=self.get_chunk_context
        )

        # Temporal Filter
        self.tools["temporal_filter"] = Tool(
            name="temporal_filter",
            description="Filter chunks by date range. "
                       "Use this when the query involves specific time periods.",
            parameters={
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk IDs to filter",
                    "required": True
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                    "required": False
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                    "required": False
                }
            },
            function=self.temporal_filter
        )

        # Get Attachment Content
        self.tools["get_attachment_content"] = Tool(
            name="get_attachment_content",
            description="Get extracted text content from an email attachment. "
                       "Use this when you need details from attached documents.",
            parameters={
                "attachment_id": {
                    "type": "string",
                    "description": "The attachment ID",
                    "required": True
                }
            },
            function=self.get_attachment_content
        )

    def _load_graph(self):
        """Lazy load the knowledge graph."""
        if self._graph is None:
            from gold.graph_builder import GraphBuilder
            builder = GraphBuilder("", str(self.gold_path))
            self._graph = builder.load()
        return self._graph

    def _load_path_indexer(self):
        """Lazy load the path indexer."""
        if self._path_indexer is None:
            from gold.path_indexer import PathIndexer
            self._path_indexer = PathIndexer(self._load_graph(), str(self.gold_path))
            try:
                self._path_indexer.load()
                logger.info("Loaded pre-computed path index")
            except FileNotFoundError:
                # Don't build full index - use on-demand path finding instead
                logger.info("No pre-computed path index, will use on-demand path finding")
        return self._path_indexer

    def _load_embeddings(self):
        """Lazy load chunk embeddings."""
        if self._chunk_embeddings is None:
            from gold.embedding_generator import EmbeddingGenerator
            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)
            self._embedding_generator = generator
            try:
                self._chunk_ids, self._chunk_embeddings = generator.load_embeddings("chunks")
            except FileNotFoundError:
                logger.warning("Chunk embeddings not found")
                self._chunk_ids = []
                self._chunk_embeddings = None
        return self._chunk_ids, self._chunk_embeddings

    def _load_entity_embeddings(self):
        """Lazy load entity embeddings and ID-to-name mapping."""
        if self._entity_embeddings is None:
            from gold.embedding_generator import EmbeddingGenerator
            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)
            try:
                self._entity_ids, self._entity_embeddings = generator.load_embeddings("entities")
                # Build ID → name mapping from graph nodes
                self._entity_id_to_name = {}
                nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
                if nodes_file.exists():
                    with open(nodes_file, 'r', encoding='utf-8') as f:
                        nodes = json.load(f)
                    for node_id, node_data in nodes.items():
                        self._entity_id_to_name[node_id] = node_data.get("name", node_id)
                logger.info(f"Loaded {len(self._entity_ids)} entity embeddings for node retrieval")
            except FileNotFoundError:
                logger.warning("Entity embeddings not found — falling back to string matching")
                self._entity_ids = []
                self._entity_embeddings = None
                self._entity_id_to_name = {}
        return self._entity_ids, self._entity_embeddings

    def node_retrieval(self, keywords: List[str], top_n: int = 10) -> List[str]:
        """
        PathRAG Node Retrieval: dense vector matching of keywords against entity embeddings.

        Per the PathRAG paper (Stage 1):
        1. Encode keywords using the same embedding model
        2. Cosine similarity against pre-computed entity embeddings
        3. Return top-N entity names

        Args:
            keywords: Extracted keywords from query
            top_n: Maximum number of entities to return

        Returns:
            List of entity names (graph node names)
        """
        entity_ids, entity_embeddings = self._load_entity_embeddings()
        if entity_embeddings is None or len(entity_ids) == 0:
            return []

        try:
            import numpy as np

            # Get or create embedding generator for query encoding
            if self._embedding_generator is None:
                from gold.embedding_generator import EmbeddingGenerator
                self._embedding_generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)

            generator = self._embedding_generator

            # Embed all keywords in a batch
            keyword_embeddings = generator.embed_batch(keywords)
            valid_embeddings = [e for e in keyword_embeddings if e is not None]
            if not valid_embeddings:
                return []

            keyword_matrix = np.array(valid_embeddings)

            # Normalize for cosine similarity
            keyword_norms = np.linalg.norm(keyword_matrix, axis=1, keepdims=True)
            keyword_norms[keyword_norms == 0] = 1
            keyword_matrix_norm = keyword_matrix / keyword_norms

            entity_norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
            entity_norms[entity_norms == 0] = 1
            entity_matrix_norm = entity_embeddings / entity_norms

            # Compute similarity: each keyword against all entities
            # Shape: (num_keywords, num_entities)
            similarity_matrix = keyword_matrix_norm @ entity_matrix_norm.T

            # For each entity, take the max similarity across all keywords
            max_similarities = similarity_matrix.max(axis=0)

            # Get top-N entity indices
            top_indices = np.argsort(max_similarities)[::-1][:top_n]

            # Map back to entity names
            matched_names = []
            for idx in top_indices:
                if max_similarities[idx] < 0.3:  # Minimum similarity threshold
                    break
                entity_id = entity_ids[idx]
                name = self._entity_id_to_name.get(entity_id, entity_id)
                if name not in matched_names:
                    matched_names.append(name)

            logger.info(
                f"Node retrieval: {len(keywords)} keywords → {len(matched_names)} entities "
                f"(top score: {max_similarities[top_indices[0]]:.3f})"
            )
            return matched_names

        except Exception as e:
            logger.error(f"Node retrieval failed: {e}")
            return []

    def pathrag_search(
        self,
        entities: List[str],
        max_paths: int = 5
    ) -> ToolResult:
        """
        Find paths between entities using PathRAG algorithms.

        Uses flow-based pruning from the PathRAG paper:
        1. DFS path finding (up to 3 hops)
        2. BFS weighted paths for flow-based pruning
        3. Path-to-text conversion for LLM prompting
        """
        start_time = datetime.now()

        try:
            from .pathrag_retriever import PathRAGRetriever, PathRAGConfig
            import asyncio

            # Initialize PathRAG retriever
            config = PathRAGConfig(
                max_hops=3,
                flow_threshold=0.3,
                flow_alpha=0.8,
                top_k_paths=max_paths
            )
            retriever = PathRAGRetriever(str(self.gold_path), config)

            # Find entity IDs
            entity_ids = retriever.find_entity_ids_by_name(entities)

            if len(entity_ids) < 2:
                return ToolResult(
                    tool_name="pathrag_search",
                    success=False,
                    data=[],
                    message=f"Could not find enough entities. Found: {len(entity_ids)}"
                )

            # Run async retrieval
            loop = asyncio.new_event_loop()
            try:
                path_results = loop.run_until_complete(
                    retriever.retrieve(entity_ids, max_paths)
                )
            finally:
                loop.close()

            # Format results
            results = []
            for pr in path_results:
                results.append({
                    "path_id": f"path_{hash(tuple(pr.path)) & 0xFFFFFFFF:08x}",
                    "description": pr.natural_language,
                    "path": pr.path_names,
                    "path_types": pr.path_types,
                    "hop_count": pr.hop_count,
                    "weight": pr.weight,
                    "evidence_chunks": pr.evidence_chunks[:5]
                })

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="pathrag_search",
                success=True,
                data=results,
                message=f"Found {len(results)} paths (flow-based pruning)",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"PathRAG search failed: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(
                tool_name="pathrag_search",
                success=False,
                data=[],
                message=str(e)
            )

    def graphrag_search(
        self,
        query: str,
        level: int = 0,
        top_k: int = 3
    ) -> ToolResult:
        """Search communities using GraphRAG."""
        start_time = datetime.now()

        try:
            # Load community summaries
            communities_path = self.gold_path / "communities" / f"level_{level}"

            if not communities_path.exists():
                return ToolResult(
                    tool_name="graphrag_search",
                    success=False,
                    data=[],
                    message=f"Community level {level} not found"
                )

            # Load all communities at this level
            communities = []
            for comm_file in communities_path.glob("*.json"):
                with open(comm_file, 'r', encoding='utf-8') as f:
                    communities.append(json.load(f))

            if not communities:
                return ToolResult(
                    tool_name="graphrag_search",
                    success=False,
                    data=[],
                    message="No communities found"
                )

            # Simple keyword matching (could be enhanced with embeddings)
            query_terms = set(query.lower().split())
            scored_communities = []

            for comm in communities:
                summary = comm.get("summary", "").lower()
                topics = [t.lower() for t in comm.get("key_topics", [])]
                entities = [e["name"].lower() for e in comm.get("key_entities", [])]

                # Score based on term overlap
                score = 0
                for term in query_terms:
                    if term in summary:
                        score += 2
                    if any(term in t for t in topics):
                        score += 3
                    if any(term in e for e in entities):
                        score += 2

                if score > 0:
                    scored_communities.append((score, comm))

            # Sort by score
            scored_communities.sort(key=lambda x: x[0], reverse=True)

            # Return top-k
            results = []
            for score, comm in scored_communities[:top_k]:
                results.append({
                    "community_id": comm.get("community_id"),
                    "level": comm.get("level"),
                    "summary": comm.get("summary"),
                    "key_topics": comm.get("key_topics", []),
                    "key_entities": comm.get("key_entities", [])[:5],
                    "source_chunk_ids": comm.get("source_chunk_ids", []),
                    "relevance_score": score
                })

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="graphrag_search",
                success=True,
                data=results,
                message=f"Found {len(results)} relevant communities",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return ToolResult(
                tool_name="graphrag_search",
                success=False,
                data=[],
                message=str(e)
            )

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        filter_thread: Optional[str] = None
    ) -> ToolResult:
        """Semantic similarity search over chunks."""
        start_time = datetime.now()

        try:
            chunk_ids, chunk_embeddings = self._load_embeddings()
            generator = self._embedding_generator

            if chunk_embeddings is None or len(chunk_ids) == 0:
                return ToolResult(
                    tool_name="vector_search",
                    success=False,
                    data=[],
                    message="Chunk embeddings not available"
                )

            # Search
            results = generator.similarity_search(
                query, chunk_embeddings, chunk_ids, top_k=top_k * 2  # Get more for filtering
            )

            # Load chunk details
            detailed_results = []
            for chunk_id, score in results:
                if filter_thread:
                    # Would need to load chunk to check thread
                    pass

                # Load chunk data
                chunk_data = self._load_chunk(chunk_id)
                if chunk_data:
                    detailed_results.append({
                        "chunk_id": chunk_id,
                        "similarity_score": score,
                        "text": chunk_data.get("text_anonymized", "")[:500],
                        "thread_id": chunk_data.get("thread_id"),
                        "thread_subject": chunk_data.get("thread_subject"),
                        "source_type": chunk_data.get("source_type", "email"),
                        "source_attachment_filename": chunk_data.get("source_attachment_filename", ""),
                        "has_attachments": chunk_data.get("has_attachments", False)
                    })

                if len(detailed_results) >= top_k:
                    break

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="vector_search",
                success=True,
                data=detailed_results,
                message=f"Found {len(detailed_results)} similar chunks",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return ToolResult(
                tool_name="vector_search",
                success=False,
                data=[],
                message=str(e)
            )

    def list_entities(self, entity_type: str) -> ToolResult:
        """List all entities of a given type from the knowledge graph."""
        start_time = datetime.now()

        try:
            # Load nodes directly from JSON
            nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
            if not nodes_file.exists():
                return ToolResult(
                    tool_name="list_entities",
                    success=False,
                    data=[],
                    message="Knowledge graph nodes not found"
                )

            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes = json.load(f)

            entity_type_upper = entity_type.upper()
            entities = []

            for node_id, node_data in nodes.items():
                node_type = node_data.get("node_type", node_data.get("type", ""))
                if node_type == entity_type_upper:
                    name = node_data.get("name", node_id)
                    mention_count = node_data.get("mention_count", 0)
                    entities.append({
                        "name": name,
                        "type": entity_type_upper,
                        "connections": mention_count,
                    })

            # Sort by mention count (most mentioned first)
            entities.sort(key=lambda x: x["connections"], reverse=True)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="list_entities",
                success=True,
                data=entities,
                message=f"Found {len(entities)} entities of type {entity_type_upper}",
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="list_entities",
                success=False,
                data=[],
                message=str(e)
            )

    def list_entity_types(self) -> ToolResult:
        """Discover what entity types exist in the knowledge graph."""
        start_time = datetime.now()

        try:
            nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
            if not nodes_file.exists():
                return ToolResult(
                    tool_name="list_entity_types",
                    success=False,
                    data=[],
                    message="Knowledge graph nodes not found"
                )

            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes = json.load(f)

            # Count entities per type
            type_counts = {}
            for node_id, node_data in nodes.items():
                node_type = node_data.get("node_type", node_data.get("type", "UNKNOWN"))
                type_counts[node_type] = type_counts.get(node_type, 0) + 1

            # Sort by count descending
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            results = [{"type": t, "count": c} for t, c in sorted_types]

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="list_entity_types",
                success=True,
                data=results,
                message=f"Found {len(results)} entity types: " + ", ".join(f"{t}({c})" for t, c in sorted_types),
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name="list_entity_types",
                success=False,
                data=[],
                message=str(e)
            )

    def entity_lookup(self, entity_name: str) -> ToolResult:
        """Look up entity details and relationships."""
        start_time = datetime.now()

        try:
            graph = self._load_graph()

            # Find matching entity
            matching_nodes = []
            for node_id, node in graph.nodes.items():
                if entity_name.lower() in node.name.lower():
                    matching_nodes.append(node)

            if not matching_nodes:
                return ToolResult(
                    tool_name="entity_lookup",
                    success=False,
                    data=None,
                    message=f"Entity '{entity_name}' not found"
                )

            # Get details for best match
            node = matching_nodes[0]

            # Get relationships
            outgoing = graph.get_edges_from(node.node_id)
            incoming = graph.get_edges_to(node.node_id)

            relationships = []
            for edge in outgoing[:10]:
                target = graph.get_node(edge.target_id)
                if target:
                    relationships.append({
                        "direction": "outgoing",
                        "type": edge.edge_type,
                        "target": target.name,
                        "target_type": target.node_type
                    })

            for edge in incoming[:10]:
                source = graph.get_node(edge.source_id)
                if source:
                    relationships.append({
                        "direction": "incoming",
                        "type": edge.edge_type,
                        "source": source.name,
                        "source_type": source.node_type
                    })

            result = {
                "node_id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "mention_count": node.mention_count,
                "properties": node.properties,
                "relationships": relationships,
                "source_chunks": node.source_chunks[:5]
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="entity_lookup",
                success=True,
                data=result,
                message=f"Found entity: {node.name}",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Entity lookup failed: {e}")
            return ToolResult(
                tool_name="entity_lookup",
                success=False,
                data=None,
                message=str(e)
            )

    def get_chunk_context(self, chunk_id: str) -> ToolResult:
        """Get full context for a chunk."""
        start_time = datetime.now()

        try:
            chunk_data = self._load_chunk(chunk_id)

            if not chunk_data:
                return ToolResult(
                    tool_name="get_chunk_context",
                    success=False,
                    data=None,
                    message=f"Chunk '{chunk_id}' not found"
                )

            result = {
                "chunk_id": chunk_id,
                "text": chunk_data.get("text_anonymized", chunk_data.get("text_original", "")),
                "thread_id": chunk_data.get("thread_id"),
                "thread_subject": chunk_data.get("thread_subject"),
                "email_position": chunk_data.get("email_position"),
                "thread_participants": chunk_data.get("thread_participants", []),
                "has_attachments": chunk_data.get("has_attachments", False),
                "attachment_filenames": chunk_data.get("attachment_filenames", []),
                "source_type": chunk_data.get("source_type", "email"),
                "source_attachment_filename": chunk_data.get("source_attachment_filename", ""),
                "kg_entities": chunk_data.get("kg_entities", []),
                "kg_relationships": chunk_data.get("kg_relationships", [])
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="get_chunk_context",
                success=True,
                data=result,
                message="Chunk context retrieved",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Get chunk context failed: {e}")
            return ToolResult(
                tool_name="get_chunk_context",
                success=False,
                data=None,
                message=str(e)
            )

    def temporal_filter(
        self,
        chunk_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> ToolResult:
        """Filter chunks by date range."""
        start_time = datetime.now()

        try:
            filtered = []

            for chunk_id in chunk_ids:
                chunk_data = self._load_chunk(chunk_id)
                if not chunk_data:
                    continue

                # Extract date from chunk (would need to parse from text or metadata)
                # For now, use processing_time or extract from thread_id
                chunk_date = chunk_data.get("processing_time", "")[:10]

                include = True
                if start_date and chunk_date < start_date:
                    include = False
                if end_date and chunk_date > end_date:
                    include = False

                if include:
                    filtered.append(chunk_id)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="temporal_filter",
                success=True,
                data=filtered,
                message=f"Filtered to {len(filtered)} chunks in date range",
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Temporal filter failed: {e}")
            return ToolResult(
                tool_name="temporal_filter",
                success=False,
                data=[],
                message=str(e)
            )

    def get_attachment_content(self, attachment_id: str) -> ToolResult:
        """Get extracted text from an attachment."""
        start_time = datetime.now()

        try:
            if not self.silver_path:
                return ToolResult(
                    tool_name="get_attachment_content",
                    success=False,
                    data=None,
                    message="Silver path not configured"
                )

            # Check attachment content cache
            content_path = self.silver_path / "attachment_content" / f"{attachment_id}.json"

            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    content_data = json.load(f)

                result = {
                    "attachment_id": attachment_id,
                    "filename": content_data.get("filename"),
                    "extracted_text": content_data.get("extracted_text", "")[:2000],
                    "text_length": content_data.get("text_length", 0),
                    "extraction_method": content_data.get("extraction_method")
                }

                execution_time = (datetime.now() - start_time).total_seconds()

                return ToolResult(
                    tool_name="get_attachment_content",
                    success=True,
                    data=result,
                    message="Attachment content retrieved",
                    execution_time=execution_time
                )

            return ToolResult(
                tool_name="get_attachment_content",
                success=False,
                data=None,
                message=f"Attachment content not found: {attachment_id}"
            )

        except Exception as e:
            logger.error(f"Get attachment content failed: {e}")
            return ToolResult(
                tool_name="get_attachment_content",
                success=False,
                data=None,
                message=str(e)
            )

    def _load_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Load chunk data from Silver layer."""
        if not self.silver_path:
            return None

        # Try known chunk directories first
        for pattern in ["technical/thread_chunks", "technical/email_chunks", "technical/attachment_chunks"]:
            chunk_path = self.silver_path / pattern / f"{chunk_id}.json"
            if chunk_path.exists():
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        # Try glob search
        for chunk_file in self.silver_path.glob(f"**/{chunk_id}.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with arguments."""
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                data=None,
                message=f"Tool '{name}' not found"
            )

        return tool.function(**kwargs)
