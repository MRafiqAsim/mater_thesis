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
        silver_path: Optional[str] = None
    ):
        """
        Initialize the retrieval toolkit.

        Args:
            gold_path: Path to Gold layer with graph and indexes
            silver_path: Path to Silver layer with chunks (optional)
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None

        # Lazy-loaded components
        self._graph = None
        self._path_indexer = None
        self._community_detector = None
        self._embedding_generator = None
        self._chunk_embeddings = None
        self._chunk_ids = None

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
            from src.graph.graph_builder import GraphBuilder
            builder = GraphBuilder("", str(self.gold_path))
            self._graph = builder.load()
        return self._graph

    def _load_path_indexer(self):
        """Lazy load the path indexer."""
        if self._path_indexer is None:
            from src.graph.path_indexer import PathIndexer
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
            from src.graph.embedding_generator import EmbeddingGenerator
            generator = EmbeddingGenerator(str(self.gold_path))
            try:
                self._chunk_ids, self._chunk_embeddings = generator.load_embeddings("chunks")
            except FileNotFoundError:
                logger.warning("Chunk embeddings not found")
                self._chunk_ids = []
                self._chunk_embeddings = None
        return self._chunk_ids, self._chunk_embeddings

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
            from src.graph.embedding_generator import EmbeddingGenerator

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
            from src.graph.embedding_generator import EmbeddingGenerator

            generator = EmbeddingGenerator(str(self.gold_path))
            chunk_ids, chunk_embeddings = self._load_embeddings()

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
                        "thread_subject": chunk_data.get("thread_subject"),
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

        # Try thread chunks first
        for pattern in ["thread_chunks", "individual_chunks"]:
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
