"""
Knowledge Graph Storage Module
==============================
Azure Cosmos DB Gremlin API integration for knowledge graph storage.

Features:
- Entity (vertex) storage with properties
- Relationship (edge) storage with weights
- Graph traversal queries
- Batch operations with retry logic
- Graph statistics and analytics

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """Configuration for Cosmos DB Gremlin."""
    endpoint: str
    database: str = "knowledge-graph"
    graph: str = "entities"
    partition_key: str = "/type"
    batch_size: int = 100


class CosmosGraphStore:
    """
    Azure Cosmos DB Gremlin graph store.

    Usage:
        store = CosmosGraphStore(endpoint, key, config)
        store.add_entity(entity_dict)
        store.add_relationship(rel_dict)
    """

    def __init__(
        self,
        endpoint: str,
        key: str,
        config: Optional[GraphConfig] = None
    ):
        """
        Initialize Cosmos DB Gremlin connection.

        Args:
            endpoint: Cosmos DB Gremlin endpoint (wss://...)
            key: Cosmos DB access key
            config: Graph configuration
        """
        self.config = config or GraphConfig(endpoint=endpoint)
        self.key = key

        from gremlin_python.driver import client, serializer

        self.client = client.Client(
            endpoint,
            'g',
            username=f"/dbs/{self.config.database}/colls/{self.config.graph}",
            password=key,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )

        logger.info(f"Connected to Cosmos DB Gremlin: {self.config.database}/{self.config.graph}")

    def close(self):
        """Close the connection."""
        if self.client:
            self.client.close()

    # ============================================
    # Entity (Vertex) Operations
    # ============================================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def add_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Add or update an entity (vertex) in the graph.

        Args:
            entity: Entity dictionary with keys:
                   - id: Unique entity ID
                   - name: Entity name
                   - type: Entity type (PERSON, ORGANIZATION, etc.)
                   - description: Entity description
                   - properties: Additional properties

        Returns:
            True if successful
        """
        entity_id = entity.get("id")
        entity_type = entity.get("type", "UNKNOWN")
        name = entity.get("name", "")
        description = entity.get("description", "")
        mention_count = entity.get("mention_count", 1)

        # Upsert query
        query = """
        g.V().has('entity', 'id', entity_id).fold().coalesce(
            unfold(),
            addV('entity')
                .property('id', entity_id)
                .property('pk', entity_type)
        )
        .property('name', name)
        .property('type', entity_type)
        .property('description', description)
        .property('mention_count', mention_count)
        """

        # Add additional properties
        properties = entity.get("properties", {})
        for key, value in properties.items():
            if value is not None:
                query += f".property('{key}', {repr(value)})"

        try:
            self.client.submit(query, {
                'entity_id': entity_id,
                'entity_type': entity_type,
                'name': name,
                'description': description,
                'mention_count': mention_count,
            }).all().result()
            return True
        except Exception as e:
            logger.error(f"Failed to add entity {entity_id}: {e}")
            raise

    def add_entities_batch(self, entities: List[Dict[str, Any]]) -> tuple:
        """
        Add multiple entities in batch.

        Returns:
            Tuple of (success_count, error_count)
        """
        success = 0
        errors = 0

        for entity in entities:
            try:
                self.add_entity(entity)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to add entity {entity.get('id')}: {e}")
                errors += 1

        return success, errors

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        query = "g.V().has('entity', 'id', entity_id).valueMap(true)"

        try:
            result = self.client.submit(query, {'entity_id': entity_id}).all().result()
            if result:
                return self._parse_vertex(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
        query = "g.V().has('entity', 'type', entity_type).limit(limit).valueMap(true)"

        try:
            results = self.client.submit(query, {
                'entity_type': entity_type,
                'limit': limit
            }).all().result()
            return [self._parse_vertex(r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get entities by type {entity_type}: {e}")
            return []

    # ============================================
    # Relationship (Edge) Operations
    # ============================================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def add_relationship(self, relationship: Dict[str, Any]) -> bool:
        """
        Add or update a relationship (edge) in the graph.

        Args:
            relationship: Relationship dictionary with keys:
                         - id: Unique relationship ID
                         - source_id: Source entity ID
                         - target_id: Target entity ID
                         - type: Relationship type
                         - description: Relationship description
                         - strength: Relationship strength (0.0-1.0)

        Returns:
            True if successful
        """
        rel_id = relationship.get("id")
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        rel_type = relationship.get("type", "RELATED_TO")
        description = relationship.get("description", "")
        strength = relationship.get("strength", 1.0)

        # Check if edge exists, create if not
        query = """
        g.V().has('entity', 'id', source_id).as('source')
         .V().has('entity', 'id', target_id).as('target')
         .coalesce(
             select('source').outE(rel_type).where(inV().as('target')),
             select('source').addE(rel_type).to(select('target'))
                 .property('id', rel_id)
                 .property('description', description)
                 .property('strength', strength)
         )
        """

        try:
            self.client.submit(query, {
                'source_id': source_id,
                'target_id': target_id,
                'rel_type': rel_type,
                'rel_id': rel_id,
                'description': description,
                'strength': strength,
            }).all().result()
            return True
        except Exception as e:
            logger.error(f"Failed to add relationship {rel_id}: {e}")
            raise

    def add_relationships_batch(self, relationships: List[Dict[str, Any]]) -> tuple:
        """
        Add multiple relationships in batch.

        Returns:
            Tuple of (success_count, error_count)
        """
        success = 0
        errors = 0

        for rel in relationships:
            try:
                self.add_relationship(rel)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to add relationship {rel.get('id')}: {e}")
                errors += 1

        return success, errors

    # ============================================
    # Graph Traversal Queries
    # ============================================

    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities.

        Args:
            entity_id: Starting entity ID
            max_hops: Maximum traversal depth
            direction: 'out', 'in', or 'both'

        Returns:
            List of neighbor entities with relationship info
        """
        if direction == "out":
            traverse = ".out()"
        elif direction == "in":
            traverse = ".in()"
        else:
            traverse = ".both()"

        # Build hop query
        hop_query = traverse
        for _ in range(max_hops - 1):
            hop_query += traverse

        query = f"g.V().has('entity', 'id', entity_id){hop_query}.dedup().valueMap(true)"

        try:
            results = self.client.submit(query, {'entity_id': entity_id}).all().result()
            return [self._parse_vertex(r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get neighbors for {entity_id}: {e}")
            return []

    def get_relationships_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships involving an entity."""
        query = """
        g.V().has('entity', 'id', entity_id)
         .bothE()
         .project('id', 'type', 'source', 'target', 'description', 'strength')
         .by('id')
         .by(label)
         .by(outV().values('id'))
         .by(inV().values('id'))
         .by('description')
         .by('strength')
        """

        try:
            results = self.client.submit(query, {'entity_id': entity_id}).all().result()
            return results
        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")
            return []

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities."""
        query = """
        g.V().has('entity', 'id', source_id)
         .repeat(both().simplePath())
         .until(has('entity', 'id', target_id).or().loops().is(max_depth))
         .has('entity', 'id', target_id)
         .path()
         .limit(1)
        """

        try:
            results = self.client.submit(query, {
                'source_id': source_id,
                'target_id': target_id,
                'max_depth': max_depth,
            }).all().result()
            if results:
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Failed to find path from {source_id} to {target_id}: {e}")
            return None

    # ============================================
    # Graph Statistics
    # ============================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {}

        # Vertex count
        try:
            result = self.client.submit("g.V().count()").all().result()
            stats["vertex_count"] = result[0] if result else 0
        except Exception:
            stats["vertex_count"] = -1

        # Edge count
        try:
            result = self.client.submit("g.E().count()").all().result()
            stats["edge_count"] = result[0] if result else 0
        except Exception:
            stats["edge_count"] = -1

        # Vertex count by type
        try:
            result = self.client.submit(
                "g.V().groupCount().by('type')"
            ).all().result()
            stats["vertices_by_type"] = result[0] if result else {}
        except Exception:
            stats["vertices_by_type"] = {}

        # Edge count by type
        try:
            result = self.client.submit(
                "g.E().groupCount().by(label)"
            ).all().result()
            stats["edges_by_type"] = result[0] if result else {}
        except Exception:
            stats["edges_by_type"] = {}

        return stats

    def get_all_vertices(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all vertices (for community detection)."""
        query = "g.V().limit(limit).valueMap(true)"

        try:
            results = self.client.submit(query, {'limit': limit}).all().result()
            return [self._parse_vertex(r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get all vertices: {e}")
            return []

    def get_all_edges(self, limit: int = 50000) -> List[Dict[str, Any]]:
        """Get all edges (for community detection)."""
        query = """
        g.E().limit(limit)
         .project('source', 'target', 'type', 'strength')
         .by(outV().values('id'))
         .by(inV().values('id'))
         .by(label)
         .by(coalesce(values('strength'), constant(1.0)))
        """

        try:
            results = self.client.submit(query, {'limit': limit}).all().result()
            return results
        except Exception as e:
            logger.error(f"Failed to get all edges: {e}")
            return []

    # ============================================
    # Utility Methods
    # ============================================

    def _parse_vertex(self, vertex_map: Dict) -> Dict[str, Any]:
        """Parse Gremlin vertex map to simple dict."""
        result = {}
        for key, value in vertex_map.items():
            if isinstance(value, list) and len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value
        return result

    def clear_graph(self) -> bool:
        """Clear all vertices and edges (use with caution!)."""
        try:
            self.client.submit("g.V().drop()").all().result()
            logger.warning("Graph cleared!")
            return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False


# ============================================
# In-Memory Graph Store (Alternative)
# ============================================

class InMemoryGraphStore:
    """
    In-memory graph store for development and testing.

    Can be exported to NetworkX or igraph for analysis.
    """

    def __init__(self):
        self.vertices: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_entity(self, entity: Dict[str, Any]) -> bool:
        entity_id = entity.get("id")
        self.vertices[entity_id] = entity
        return True

    def add_relationship(self, relationship: Dict[str, Any]) -> bool:
        self.edges.append(relationship)
        return True

    def add_entities_batch(self, entities: List[Dict[str, Any]]) -> tuple:
        for entity in entities:
            self.add_entity(entity)
        return len(entities), 0

    def add_relationships_batch(self, relationships: List[Dict[str, Any]]) -> tuple:
        for rel in relationships:
            self.add_relationship(rel)
        return len(relationships), 0

    def get_statistics(self) -> Dict[str, Any]:
        from collections import Counter

        vertex_types = Counter(v.get("type") for v in self.vertices.values())
        edge_types = Counter(e.get("type") for e in self.edges)

        return {
            "vertex_count": len(self.vertices),
            "edge_count": len(self.edges),
            "vertices_by_type": dict(vertex_types),
            "edges_by_type": dict(edge_types),
        }

    def to_networkx(self):
        """Export to NetworkX graph."""
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes
        for vertex_id, vertex in self.vertices.items():
            G.add_node(vertex_id, **vertex)

        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge["source_id"],
                edge["target_id"],
                type=edge.get("type"),
                weight=edge.get("strength", 1.0),
            )

        return G

    def to_igraph(self):
        """Export to igraph for community detection."""
        import igraph as ig

        # Create vertex mapping
        vertex_ids = list(self.vertices.keys())
        id_to_idx = {vid: i for i, vid in enumerate(vertex_ids)}

        # Create graph
        g = ig.Graph(directed=False)
        g.add_vertices(len(vertex_ids))

        # Add vertex attributes
        g.vs["id"] = vertex_ids
        g.vs["name"] = [self.vertices[vid].get("name", "") for vid in vertex_ids]
        g.vs["type"] = [self.vertices[vid].get("type", "") for vid in vertex_ids]

        # Add edges
        edge_list = []
        weights = []
        for edge in self.edges:
            source_idx = id_to_idx.get(edge["source_id"])
            target_idx = id_to_idx.get(edge["target_id"])
            if source_idx is not None and target_idx is not None:
                edge_list.append((source_idx, target_idx))
                weights.append(edge.get("strength", 1.0))

        g.add_edges(edge_list)
        g.es["weight"] = weights

        return g


# Export
__all__ = [
    'CosmosGraphStore',
    'InMemoryGraphStore',
    'GraphConfig',
]
