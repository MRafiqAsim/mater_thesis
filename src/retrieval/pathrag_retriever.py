"""
PathRAG Retriever Integration

Integrates the official PathRAG algorithms with our pipeline.
Uses PathRAG's flow-based path pruning for query-time path finding.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class PathRAGConfig:
    """Configuration for PathRAG retrieval."""
    max_hops: int = 3  # Maximum path length (1-3 hops)
    flow_threshold: float = 0.3  # Threshold for flow-based pruning
    flow_alpha: float = 0.8  # Alpha parameter for BFS weighted paths
    top_k_entities: int = 10  # Top-k entities from vector search
    top_k_paths: int = 15  # Maximum paths to return
    max_token_for_context: int = 4000


@dataclass
class PathResult:
    """Result from path-based retrieval."""
    path: List[str]  # Node IDs in path
    path_names: List[str]  # Node names
    path_types: List[str]  # Node types
    edges: List[Dict[str, Any]]  # Edge data
    weight: float  # Path weight from flow-based pruning
    hop_count: int
    natural_language: str  # Path converted to text
    evidence_chunks: List[str]  # Source chunks


class PathRAGRetriever:
    """
    PathRAG retriever using flow-based path pruning.

    Implements the key algorithms from the PathRAG paper:
    1. DFS path finding (up to 3 hops)
    2. Flow-based pruning with BFS weighted paths
    3. Path-to-text conversion for LLM prompting
    """

    def __init__(
        self,
        gold_path: str,
        config: Optional[PathRAGConfig] = None
    ):
        self.gold_path = Path(gold_path)
        self.config = config or PathRAGConfig()

        # Lazy-loaded components
        self._graph = None
        self._nx_graph = None
        self._embeddings = None

        logger.info("PathRAGRetriever initialized")

    def _load_graph(self):
        """Load knowledge graph."""
        if self._graph is None:
            from src.graph.graph_builder import GraphBuilder
            builder = GraphBuilder("", str(self.gold_path))
            self._graph = builder.load()
            logger.info(f"Loaded graph: {len(self._graph.nodes)} nodes, {len(self._graph.edges)} edges")
        return self._graph

    def _build_nx_graph(self) -> nx.Graph:
        """Build NetworkX graph from knowledge graph."""
        if self._nx_graph is None:
            graph = self._load_graph()
            self._nx_graph = nx.Graph()

            # Add nodes
            for node_id, node in graph.nodes.items():
                self._nx_graph.add_node(
                    node_id,
                    name=node.name,
                    type=node.node_type,
                    source_chunks=node.source_chunks
                )

            # Add edges
            for edge_id, edge in graph.edges.items():
                self._nx_graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    description=edge.properties.get("description", ""),
                    keywords=edge.properties.get("keywords", "")
                )

            logger.info(f"Built NetworkX graph: {self._nx_graph.number_of_nodes()} nodes, {self._nx_graph.number_of_edges()} edges")

        return self._nx_graph

    async def find_paths_between_entities(
        self,
        source_entities: List[str]
    ) -> Tuple[Dict, Dict, List, List, List]:
        """
        Find paths between entities using DFS (PathRAG algorithm).

        Args:
            source_entities: List of entity node IDs

        Returns:
            Tuple of (result_dict, path_stats, one_hop, two_hop, three_hop paths)
        """
        G = self._build_nx_graph()

        result = defaultdict(lambda: {"paths": [], "edges": set()})
        path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}
        one_hop_paths = []
        two_hop_paths = []
        three_hop_paths = []

        def dfs(current: str, target: str, path: List[str], depth: int):
            """DFS to find paths up to max_hops."""
            if depth > self.config.max_hops:
                return
            if current == target:
                result[(path[0], target)]["paths"].append(list(path))
                for u, v in zip(path[:-1], path[1:]):
                    result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
                if depth == 1:
                    path_stats["1-hop"] += 1
                    one_hop_paths.append(list(path))
                elif depth == 2:
                    path_stats["2-hop"] += 1
                    two_hop_paths.append(list(path))
                elif depth == 3:
                    path_stats["3-hop"] += 1
                    three_hop_paths.append(list(path))
                return

            if current not in G:
                return

            for neighbor in G.neighbors(current):
                if neighbor not in path:
                    dfs(neighbor, target, path + [neighbor], depth + 1)

        # Find paths between all entity pairs
        for node1 in source_entities:
            for node2 in source_entities:
                if node1 != node2:
                    dfs(node1, node2, [node1], 0)

        # Convert edges to lists
        for key in result:
            result[key]["edges"] = list(result[key]["edges"])

        return dict(result), path_stats, one_hop_paths, two_hop_paths, three_hop_paths

    def bfs_weighted_paths(
        self,
        paths: List[List[str]],
        source: str,
        target: str
    ) -> List[Tuple[List[str], float]]:
        """
        Flow-based path pruning using BFS weighted paths.

        This is the key PathRAG algorithm that prunes redundant paths
        based on edge flow weights.
        """
        threshold = self.config.flow_threshold
        alpha = self.config.flow_alpha

        edge_weights = defaultdict(float)
        follow_dict = {}

        # Build follow dictionary from paths
        for p in paths:
            for i in range(len(p) - 1):
                current = p[i]
                next_node = p[i + 1]
                if current in follow_dict:
                    follow_dict[current].add(next_node)
                else:
                    follow_dict[current] = {next_node}

        if source not in follow_dict:
            return [(p, 1.0) for p in paths]  # Return all paths if no flow

        results = []

        # BFS to compute edge weights
        for neighbor in follow_dict[source]:
            edge_weights[(source, neighbor)] += 1 / len(follow_dict[source])

            if neighbor == target:
                results.append([source, neighbor])
                continue

            if edge_weights[(source, neighbor)] > threshold and neighbor in follow_dict:
                for second_neighbor in follow_dict[neighbor]:
                    weight = edge_weights[(source, neighbor)] * alpha / len(follow_dict[neighbor])
                    edge_weights[(neighbor, second_neighbor)] += weight

                    if second_neighbor == target:
                        results.append([source, neighbor, second_neighbor])
                        continue

                    if edge_weights[(neighbor, second_neighbor)] > threshold and second_neighbor in follow_dict:
                        for third_neighbor in follow_dict[second_neighbor]:
                            weight = edge_weights[(neighbor, second_neighbor)] * alpha / len(follow_dict[second_neighbor])
                            edge_weights[(second_neighbor, third_neighbor)] += weight

                            if third_neighbor == target:
                                results.append([source, neighbor, second_neighbor, third_neighbor])

        # Calculate path weights
        path_weights = []
        for p in paths:
            path_weight = 0
            for i in range(len(p) - 1):
                edge = (p[i], p[i + 1])
                path_weight += edge_weights.get(edge, 0)
            path_weights.append(path_weight / (len(p) - 1) if len(p) > 1 else 0)

        return list(zip(paths, path_weights))

    def path_to_natural_language(self, path: List[str]) -> str:
        """
        Convert a path to natural language description.

        This is how PathRAG presents paths to the LLM.
        """
        G = self._build_nx_graph()
        graph = self._load_graph()

        parts = []

        for i, node_id in enumerate(path):
            node_data = G.nodes.get(node_id, {})
            node_name = node_data.get("name", node_id)
            node_type = node_data.get("type", "UNKNOWN")

            # Get node description from knowledge graph
            kg_node = graph.get_node(node_id)
            description = ""
            if kg_node and kg_node.properties:
                description = kg_node.properties.get("description", "")[:100]

            entity_desc = f"Entity '{node_name}' ({node_type})"
            if description:
                entity_desc += f": {description}"

            if i == 0:
                parts.append(entity_desc)
            else:
                # Get edge info
                prev_id = path[i - 1]
                edge_data = G.edges.get((prev_id, node_id), G.edges.get((node_id, prev_id), {}))
                edge_type = edge_data.get("edge_type", "related_to")
                keywords = edge_data.get("keywords", "")

                connection = f" --[{edge_type}"
                if keywords:
                    connection += f": {keywords}"
                connection += f"]--> {entity_desc}"
                parts.append(connection)

        return "".join(parts)

    async def retrieve(
        self,
        entity_ids: List[str],
        max_paths: int = None
    ) -> List[PathResult]:
        """
        Main retrieval method using PathRAG algorithms.

        Args:
            entity_ids: List of entity node IDs to find paths between
            max_paths: Maximum paths to return

        Returns:
            List of PathResult objects
        """
        max_paths = max_paths or self.config.top_k_paths

        if len(entity_ids) < 2:
            logger.warning("Need at least 2 entities for path finding")
            return []

        # Step 1: Find all paths using DFS
        result, path_stats, one_hop, two_hop, three_hop = await self.find_paths_between_entities(entity_ids)

        logger.info(f"Path stats: {path_stats}")

        # Step 2: Apply flow-based pruning
        all_results = []
        for node1 in entity_ids:
            for node2 in entity_ids:
                if node1 != node2 and (node1, node2) in result:
                    paths = result[(node1, node2)]["paths"]
                    weighted_paths = self.bfs_weighted_paths(paths, node1, node2)
                    all_results.extend(weighted_paths)

        # Sort by weight (descending)
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)

        # Deduplicate paths
        seen = set()
        unique_results = []
        for path, weight in all_results:
            sorted_path = tuple(sorted(path))
            if sorted_path not in seen:
                seen.add(sorted_path)
                unique_results.append((path, weight))

        # Take top-k
        unique_results = unique_results[:max_paths]

        # Step 3: Convert to PathResult objects
        G = self._build_nx_graph()
        graph = self._load_graph()

        path_results = []
        for path, weight in unique_results:
            # Get node names and types
            path_names = []
            path_types = []
            evidence_chunks = set()

            for node_id in path:
                node_data = G.nodes.get(node_id, {})
                path_names.append(node_data.get("name", node_id))
                path_types.append(node_data.get("type", "UNKNOWN"))

                # Collect evidence chunks
                kg_node = graph.get_node(node_id)
                if kg_node:
                    evidence_chunks.update(kg_node.source_chunks)

            # Get edges
            edges = []
            for i in range(len(path) - 1):
                edge_data = G.edges.get((path[i], path[i + 1]), G.edges.get((path[i + 1], path[i]), {}))
                edges.append(dict(edge_data))

            # Convert to natural language
            nl_description = self.path_to_natural_language(path)

            path_results.append(PathResult(
                path=path,
                path_names=path_names,
                path_types=path_types,
                edges=edges,
                weight=weight,
                hop_count=len(path) - 1,
                natural_language=nl_description,
                evidence_chunks=list(evidence_chunks)[:10]
            ))

        logger.info(f"Returning {len(path_results)} paths")
        return path_results

    def find_entity_ids_by_name(self, names: List[str]) -> List[str]:
        """Find entity node IDs by name (bidirectional substring match)."""
        graph = self._load_graph()
        entity_ids = []

        for name in names:
            name_lower = name.lower()
            for node_id, node in graph.nodes.items():
                node_lower = node.name.lower()
                # Bidirectional: query in node name OR node name in query
                if name_lower in node_lower or node_lower in name_lower:
                    entity_ids.append(node_id)
                    break

        return entity_ids


# Async wrapper for synchronous use
def retrieve_paths_sync(
    gold_path: str,
    entity_names: List[str],
    max_paths: int = 15,
    config: Optional[PathRAGConfig] = None
) -> List[PathResult]:
    """
    Synchronous wrapper for path retrieval.

    Args:
        gold_path: Path to Gold layer
        entity_names: List of entity names to find paths between
        max_paths: Maximum paths to return
        config: PathRAG configuration

    Returns:
        List of PathResult objects
    """
    retriever = PathRAGRetriever(gold_path, config)
    entity_ids = retriever.find_entity_ids_by_name(entity_names)

    if len(entity_ids) < 2:
        return []

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(retriever.retrieve(entity_ids, max_paths))
    finally:
        loop.close()
