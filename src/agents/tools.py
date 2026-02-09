"""
Agent Tools Module
==================
LangChain tools for ReAct agent - vector search, graph queries, and community lookup.

Features:
- Vector similarity search tool
- Entity lookup tool
- Relationship exploration tool
- Community summary tool
- Graph traversal tool

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
import logging

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


# ============================================
# Tool Input Schemas (Pydantic)
# ============================================

class VectorSearchInput(BaseModel):
    """Input schema for vector search tool."""
    query: str = Field(description="The search query to find relevant document chunks")
    top_k: int = Field(default=5, description="Number of results to return (1-20)")


class EntityLookupInput(BaseModel):
    """Input schema for entity lookup tool."""
    entity_name: str = Field(description="Name of the entity to look up")
    include_relationships: bool = Field(
        default=True,
        description="Whether to include relationships involving this entity"
    )


class RelationshipSearchInput(BaseModel):
    """Input schema for relationship search tool."""
    entity_name: str = Field(description="Entity name to find relationships for")
    relationship_type: Optional[str] = Field(
        default=None,
        description="Filter by relationship type (e.g., WORKS_ON, REPORTS_TO)"
    )


class CommunitySearchInput(BaseModel):
    """Input schema for community search tool."""
    query: str = Field(description="Query to find relevant community summaries")
    level: Optional[int] = Field(
        default=None,
        description="Community hierarchy level (0=fine, 1=medium, 2=coarse)"
    )


class GraphTraversalInput(BaseModel):
    """Input schema for graph traversal tool."""
    start_entity: str = Field(description="Starting entity name for traversal")
    max_hops: int = Field(default=2, description="Maximum traversal depth (1-3)")
    direction: str = Field(
        default="both",
        description="Traversal direction: 'out', 'in', or 'both'"
    )


# ============================================
# Vector Search Tool
# ============================================

class VectorSearchTool(BaseTool):
    """
    Search document chunks using semantic similarity.

    Use this tool to find specific information, facts, or evidence
    from the source documents.
    """
    name: str = "vector_search"
    description: str = """Search for relevant document chunks using semantic similarity.
Use this when you need to find specific information, facts, quotes, or evidence from documents.
Input: A natural language query describing what you're looking for.
Returns: Relevant document excerpts with source information."""

    args_schema: Type[BaseModel] = VectorSearchInput

    vector_store: Any = None  # Injected at runtime
    embeddings: Any = None

    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute vector search."""
        try:
            if self.vector_store is None:
                raise ToolException("Vector store not configured")

            # Clamp top_k
            top_k = max(1, min(top_k, 20))

            # Perform search
            results = self.vector_store.similarity_search_with_score(query, k=top_k)

            if not results:
                return "No relevant documents found for this query."

            # Format results
            output = []
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source_file", "Unknown source")
                chunk_id = doc.metadata.get("chunk_id", "")
                content = doc.page_content[:500]  # Truncate for readability

                output.append(f"""
**Result {i}** (relevance: {score:.3f})
Source: {source}
---
{content}
---""")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise ToolException(f"Search failed: {str(e)}")


# ============================================
# Entity Lookup Tool
# ============================================

class EntityLookupTool(BaseTool):
    """
    Look up information about a specific entity.

    Use this tool to get details about people, organizations,
    projects, technologies, etc.
    """
    name: str = "entity_lookup"
    description: str = """Look up detailed information about a specific entity (person, organization, project, technology, etc.).
Use this when you need facts about a particular named entity.
Input: The name of the entity to look up.
Returns: Entity details including type, description, and optionally relationships."""

    args_schema: Type[BaseModel] = EntityLookupInput

    entities: Dict[str, Dict] = {}  # Injected: id -> entity dict
    entity_names: Dict[str, str] = {}  # Injected: name.lower() -> id
    relationships: List[Dict] = []  # Injected

    def _run(
        self,
        entity_name: str,
        include_relationships: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Look up entity information."""
        try:
            # Find entity by name (case-insensitive)
            entity_id = self.entity_names.get(entity_name.lower())

            if not entity_id:
                # Try partial matching
                for name, eid in self.entity_names.items():
                    if entity_name.lower() in name or name in entity_name.lower():
                        entity_id = eid
                        break

            if not entity_id:
                # List similar entities
                similar = [
                    name for name in self.entity_names.keys()
                    if any(word in name for word in entity_name.lower().split())
                ][:5]

                if similar:
                    return f"Entity '{entity_name}' not found. Did you mean: {', '.join(similar)}?"
                return f"Entity '{entity_name}' not found in the knowledge graph."

            entity = self.entities.get(entity_id, {})

            # Format entity info
            output = f"""
**{entity.get('name', entity_name)}**
- Type: {entity.get('type', 'Unknown')}
- Description: {entity.get('description', 'No description available')}
- Mentions: {entity.get('mention_count', 0)} times across documents
"""

            # Add relationships if requested
            if include_relationships:
                related = []
                for rel in self.relationships:
                    if rel.get("source_id") == entity_id:
                        related.append(
                            f"  → {rel.get('target_name', '?')} ({rel.get('type', 'RELATED_TO')})"
                        )
                    elif rel.get("target_id") == entity_id:
                        related.append(
                            f"  ← {rel.get('source_name', '?')} ({rel.get('type', 'RELATED_TO')})"
                        )

                if related:
                    output += f"\n**Relationships:**\n" + "\n".join(related[:10])
                    if len(related) > 10:
                        output += f"\n  ... and {len(related) - 10} more relationships"

            return output

        except Exception as e:
            logger.error(f"Entity lookup failed: {e}")
            raise ToolException(f"Lookup failed: {str(e)}")


# ============================================
# Relationship Search Tool
# ============================================

class RelationshipSearchTool(BaseTool):
    """
    Find relationships involving an entity.

    Use this tool to understand how entities are connected.
    """
    name: str = "relationship_search"
    description: str = """Find relationships involving a specific entity.
Use this to understand connections between people, projects, organizations, etc.
Input: Entity name and optionally a relationship type filter.
Returns: List of relationships (e.g., WORKS_ON, REPORTS_TO, PART_OF)."""

    args_schema: Type[BaseModel] = RelationshipSearchInput

    entities: Dict[str, Dict] = {}
    entity_names: Dict[str, str] = {}
    relationships: List[Dict] = []

    def _run(
        self,
        entity_name: str,
        relationship_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search for relationships."""
        try:
            # Find entity
            entity_id = self.entity_names.get(entity_name.lower())

            if not entity_id:
                for name, eid in self.entity_names.items():
                    if entity_name.lower() in name:
                        entity_id = eid
                        break

            if not entity_id:
                return f"Entity '{entity_name}' not found."

            # Find relationships
            results = []
            for rel in self.relationships:
                matches = (
                    rel.get("source_id") == entity_id or
                    rel.get("target_id") == entity_id
                )
                type_matches = (
                    relationship_type is None or
                    rel.get("type", "").upper() == relationship_type.upper()
                )

                if matches and type_matches:
                    direction = "→" if rel.get("source_id") == entity_id else "←"
                    other = rel.get("target_name") if direction == "→" else rel.get("source_name")
                    results.append({
                        "direction": direction,
                        "other": other,
                        "type": rel.get("type"),
                        "description": rel.get("description", ""),
                        "strength": rel.get("strength", 1.0)
                    })

            if not results:
                type_hint = f" of type {relationship_type}" if relationship_type else ""
                return f"No relationships{type_hint} found for '{entity_name}'."

            # Format output
            entity = self.entities.get(entity_id, {})
            output = f"**Relationships for {entity.get('name', entity_name)}:**\n\n"

            for r in sorted(results, key=lambda x: x["strength"], reverse=True)[:15]:
                output += f"- {r['direction']} **{r['other']}** ({r['type']})\n"
                if r["description"]:
                    output += f"  _{r['description']}_\n"

            if len(results) > 15:
                output += f"\n... and {len(results) - 15} more relationships"

            return output

        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            raise ToolException(f"Search failed: {str(e)}")


# ============================================
# Community Summary Tool
# ============================================

class CommunitySummaryTool(BaseTool):
    """
    Search community summaries for high-level topic information.

    Use this tool for understanding themes, overviews, and general topics.
    """
    name: str = "community_search"
    description: str = """Search community summaries to understand high-level themes and topics.
Use this for overview questions, understanding general themes, or when specific entity details aren't needed.
Input: A query describing the topic or theme you want to understand.
Returns: Relevant community summaries with key themes and entities."""

    args_schema: Type[BaseModel] = CommunitySearchInput

    community_summaries: List[Dict] = []
    community_embeddings: Any = None  # numpy array
    embeddings: Any = None  # embedding model

    def _run(
        self,
        query: str,
        level: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search community summaries."""
        try:
            import numpy as np

            if not self.community_summaries:
                return "No community summaries available."

            # Filter by level if specified
            summaries = self.community_summaries
            embeddings = self.community_embeddings

            if level is not None:
                indices = [
                    i for i, s in enumerate(summaries)
                    if s.get("level") == level
                ]
                if indices:
                    summaries = [summaries[i] for i in indices]
                    embeddings = embeddings[indices]

            # Semantic search
            query_embedding = np.array(self.embeddings.embed_query(query))

            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            top_k = 3
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Format output
            output = []
            for idx in top_indices:
                summary = summaries[idx]
                sim = similarities[idx]

                themes = summary.get("key_themes", [])
                themes_str = ", ".join(themes[:3]) if themes else "None identified"

                entities = summary.get("key_entities", [])
                entities_str = ", ".join(entities[:5]) if entities else "None identified"

                output.append(f"""
**Community {summary.get('community_id', 'Unknown')}** (Level {summary.get('level', '?')}, relevance: {sim:.3f})
Members: {summary.get('member_count', '?')} entities

{summary.get('summary', 'No summary available')}

Key Themes: {themes_str}
Key Entities: {entities_str}
---""")

            return "\n".join(output) if output else "No relevant communities found."

        except Exception as e:
            logger.error(f"Community search failed: {e}")
            raise ToolException(f"Search failed: {str(e)}")


# ============================================
# Graph Traversal Tool
# ============================================

class GraphTraversalTool(BaseTool):
    """
    Traverse the knowledge graph from a starting entity.

    Use this tool to explore connections and find paths between entities.
    """
    name: str = "graph_traversal"
    description: str = """Traverse the knowledge graph starting from an entity.
Use this to explore connections, find related entities, or understand entity neighborhoods.
Input: Starting entity name and traversal depth (1-3 hops).
Returns: Connected entities within the specified number of hops."""

    args_schema: Type[BaseModel] = GraphTraversalInput

    graph_store: Any = None
    entities: Dict[str, Dict] = {}
    entity_names: Dict[str, str] = {}

    def _run(
        self,
        start_entity: str,
        max_hops: int = 2,
        direction: str = "both",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Traverse the graph."""
        try:
            # Find entity
            entity_id = self.entity_names.get(start_entity.lower())

            if not entity_id:
                for name, eid in self.entity_names.items():
                    if start_entity.lower() in name:
                        entity_id = eid
                        break

            if not entity_id:
                return f"Entity '{start_entity}' not found."

            # Clamp max_hops
            max_hops = max(1, min(max_hops, 3))

            # Get neighbors from graph store
            if self.graph_store is None:
                return "Graph store not configured."

            neighbors = self.graph_store.get_neighbors(
                entity_id,
                max_hops=max_hops,
                direction=direction
            )

            if not neighbors:
                return f"No connected entities found within {max_hops} hops."

            # Format output
            start = self.entities.get(entity_id, {})
            output = f"**Graph traversal from {start.get('name', start_entity)}**\n"
            output += f"Direction: {direction}, Max hops: {max_hops}\n\n"

            # Group by type
            by_type = {}
            for n in neighbors:
                ntype = n.get("type", "UNKNOWN")
                if ntype not in by_type:
                    by_type[ntype] = []
                by_type[ntype].append(n)

            for ntype, entities in sorted(by_type.items()):
                output += f"\n**{ntype}** ({len(entities)}):\n"
                for e in entities[:10]:
                    output += f"  - {e.get('name', e.get('id', '?'))}\n"
                if len(entities) > 10:
                    output += f"  ... and {len(entities) - 10} more\n"

            return output

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            raise ToolException(f"Traversal failed: {str(e)}")


# ============================================
# Tool Factory
# ============================================

def create_agent_tools(
    vector_store=None,
    graph_store=None,
    entities: List[Dict] = None,
    relationships: List[Dict] = None,
    community_summaries: List[Dict] = None,
    community_embeddings=None,
    embeddings=None
) -> List[BaseTool]:
    """
    Create and configure agent tools.

    Args:
        vector_store: Vector store for document search
        graph_store: Graph store for traversal
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        community_summaries: List of community summary dicts
        community_embeddings: Pre-computed community embeddings
        embeddings: Embedding model

    Returns:
        List of configured LangChain tools
    """
    tools = []

    # Build lookup dictionaries
    entity_dict = {e["id"]: e for e in (entities or [])}
    entity_names = {e["name"].lower(): e["id"] for e in (entities or [])}

    # Vector search tool
    if vector_store:
        tool = VectorSearchTool()
        tool.vector_store = vector_store
        tool.embeddings = embeddings
        tools.append(tool)

    # Entity lookup tool
    if entities:
        tool = EntityLookupTool()
        tool.entities = entity_dict
        tool.entity_names = entity_names
        tool.relationships = relationships or []
        tools.append(tool)

    # Relationship search tool
    if relationships:
        tool = RelationshipSearchTool()
        tool.entities = entity_dict
        tool.entity_names = entity_names
        tool.relationships = relationships
        tools.append(tool)

    # Community summary tool
    if community_summaries and community_embeddings is not None:
        tool = CommunitySummaryTool()
        tool.community_summaries = community_summaries
        tool.community_embeddings = community_embeddings
        tool.embeddings = embeddings
        tools.append(tool)

    # Graph traversal tool
    if graph_store:
        tool = GraphTraversalTool()
        tool.graph_store = graph_store
        tool.entities = entity_dict
        tool.entity_names = entity_names
        tools.append(tool)

    return tools


# Export
__all__ = [
    'VectorSearchTool',
    'EntityLookupTool',
    'RelationshipSearchTool',
    'CommunitySummaryTool',
    'GraphTraversalTool',
    'create_agent_tools',
]
