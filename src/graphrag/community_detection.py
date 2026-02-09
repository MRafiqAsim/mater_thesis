"""
Community Detection Module
==========================
Hierarchical community detection using the Leiden algorithm.

Features:
- Multi-resolution community detection
- Hierarchical community structure
- Community statistics and quality metrics
- Export to various formats

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Represents a detected community."""
    id: str
    level: int  # Hierarchy level (0 = finest)
    resolution: float
    members: List[str]  # Entity IDs
    member_count: int
    internal_edges: int
    external_edges: int
    modularity_contribution: float
    parent_community: Optional[str] = None
    child_communities: List[str] = field(default_factory=list)


@dataclass
class CommunityHierarchy:
    """Hierarchical community structure."""
    levels: Dict[int, List[Community]]  # level -> communities
    entity_to_community: Dict[int, Dict[str, str]]  # level -> entity_id -> community_id
    total_communities: int
    modularity_scores: Dict[int, float]  # level -> modularity


@dataclass
class CommunityConfig:
    """Configuration for community detection."""
    # Resolution parameters for multi-level detection
    resolutions: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Leiden algorithm parameters
    n_iterations: int = 2
    random_state: int = 42

    # Filtering
    min_community_size: int = 3
    max_communities_per_level: int = 100


class LeidenCommunityDetector:
    """
    Community detection using the Leiden algorithm.

    Produces hierarchical communities at multiple resolutions.

    Usage:
        detector = LeidenCommunityDetector()
        hierarchy = detector.detect(graph)
    """

    def __init__(self, config: Optional[CommunityConfig] = None):
        """
        Initialize community detector.

        Args:
            config: Detection configuration
        """
        self.config = config or CommunityConfig()

    def detect(
        self,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> CommunityHierarchy:
        """
        Detect communities at multiple resolutions.

        Args:
            vertices: List of vertex dictionaries with 'id' key
            edges: List of edge dictionaries with 'source_id', 'target_id', 'strength' keys

        Returns:
            CommunityHierarchy with multi-level communities
        """
        import igraph as ig
        import leidenalg

        # Build igraph
        graph = self._build_igraph(vertices, edges)

        if graph.vcount() == 0:
            logger.warning("Empty graph, no communities to detect")
            return CommunityHierarchy(
                levels={},
                entity_to_community={},
                total_communities=0,
                modularity_scores={},
            )

        # Detect at each resolution
        levels = {}
        entity_to_community = {}
        modularity_scores = {}

        for level, resolution in enumerate(self.config.resolutions):
            logger.info(f"Detecting communities at resolution {resolution} (level {level})")

            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                n_iterations=self.config.n_iterations,
                seed=self.config.random_state,
            )

            # Process communities
            communities = self._process_partition(
                graph, partition, level, resolution
            )

            # Filter small communities
            communities = [c for c in communities
                          if c.member_count >= self.config.min_community_size]

            # Limit number of communities
            if len(communities) > self.config.max_communities_per_level:
                communities = sorted(
                    communities,
                    key=lambda c: c.member_count,
                    reverse=True
                )[:self.config.max_communities_per_level]

            levels[level] = communities

            # Build entity mapping
            entity_to_community[level] = {}
            for community in communities:
                for member_id in community.members:
                    entity_to_community[level][member_id] = community.id

            # Calculate modularity
            modularity_scores[level] = partition.modularity

            logger.info(f"Level {level}: {len(communities)} communities, modularity={partition.modularity:.4f}")

        # Build hierarchy relationships
        self._build_hierarchy(levels)

        total_communities = sum(len(comms) for comms in levels.values())

        return CommunityHierarchy(
            levels=levels,
            entity_to_community=entity_to_community,
            total_communities=total_communities,
            modularity_scores=modularity_scores,
        )

    def _build_igraph(
        self,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ):
        """Build igraph from vertices and edges."""
        import igraph as ig

        # Create vertex mapping
        vertex_ids = [v["id"] for v in vertices]
        id_to_idx = {vid: i for i, vid in enumerate(vertex_ids)}

        # Create graph
        g = ig.Graph(directed=False)
        g.add_vertices(len(vertex_ids))

        # Add vertex attributes
        g.vs["entity_id"] = vertex_ids
        g.vs["name"] = [v.get("name", "") for v in vertices]
        g.vs["type"] = [v.get("type", "") for v in vertices]

        # Add edges
        edge_list = []
        weights = []

        for edge in edges:
            source_idx = id_to_idx.get(edge.get("source_id") or edge.get("source"))
            target_idx = id_to_idx.get(edge.get("target_id") or edge.get("target"))

            if source_idx is not None and target_idx is not None:
                edge_list.append((source_idx, target_idx))
                weights.append(edge.get("strength", edge.get("weight", 1.0)))

        if edge_list:
            g.add_edges(edge_list)
            g.es["weight"] = weights

        logger.info(f"Built graph with {g.vcount()} vertices and {g.ecount()} edges")

        return g

    def _process_partition(
        self,
        graph,
        partition,
        level: int,
        resolution: float
    ) -> List[Community]:
        """Process Leiden partition into Community objects."""
        communities = []

        for idx, members in enumerate(partition):
            if not members:
                continue

            community_id = f"L{level}_C{idx}"

            # Get member entity IDs
            member_ids = [graph.vs[m]["entity_id"] for m in members]

            # Calculate edge statistics
            subgraph = graph.subgraph(members)
            internal_edges = subgraph.ecount()

            # External edges (approximation)
            total_degree = sum(graph.degree(members))
            external_edges = total_degree - (2 * internal_edges)

            communities.append(Community(
                id=community_id,
                level=level,
                resolution=resolution,
                members=member_ids,
                member_count=len(member_ids),
                internal_edges=internal_edges,
                external_edges=max(0, external_edges),
                modularity_contribution=0.0,  # Calculated separately if needed
            ))

        return communities

    def _build_hierarchy(self, levels: Dict[int, List[Community]]):
        """Build parent-child relationships between levels."""
        level_nums = sorted(levels.keys())

        for i in range(len(level_nums) - 1):
            fine_level = level_nums[i]
            coarse_level = level_nums[i + 1]

            fine_communities = levels[fine_level]
            coarse_communities = levels[coarse_level]

            # Build coarse community member sets
            coarse_member_sets = {
                c.id: set(c.members) for c in coarse_communities
            }

            # Find parent for each fine community
            for fine_comm in fine_communities:
                fine_members = set(fine_comm.members)

                best_parent = None
                best_overlap = 0

                for coarse_comm in coarse_communities:
                    overlap = len(fine_members & coarse_member_sets[coarse_comm.id])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = coarse_comm.id

                if best_parent:
                    fine_comm.parent_community = best_parent

                    # Add to parent's children
                    for coarse_comm in coarse_communities:
                        if coarse_comm.id == best_parent:
                            coarse_comm.child_communities.append(fine_comm.id)
                            break


class CommunityAnalyzer:
    """
    Analyze community structure and quality.
    """

    @staticmethod
    def get_community_statistics(hierarchy: CommunityHierarchy) -> Dict[str, Any]:
        """Get detailed community statistics."""
        stats = {
            "total_communities": hierarchy.total_communities,
            "levels": len(hierarchy.levels),
            "modularity_scores": hierarchy.modularity_scores,
            "by_level": {},
        }

        for level, communities in hierarchy.levels.items():
            sizes = [c.member_count for c in communities]
            stats["by_level"][level] = {
                "num_communities": len(communities),
                "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                "min_size": min(sizes) if sizes else 0,
                "max_size": max(sizes) if sizes else 0,
                "total_members": sum(sizes),
            }

        return stats

    @staticmethod
    def get_entity_community_mapping(
        hierarchy: CommunityHierarchy,
        level: int = 0
    ) -> Dict[str, str]:
        """Get entity to community mapping for a specific level."""
        return hierarchy.entity_to_community.get(level, {})

    @staticmethod
    def get_community_members(
        hierarchy: CommunityHierarchy,
        community_id: str
    ) -> Optional[List[str]]:
        """Get members of a specific community."""
        for level, communities in hierarchy.levels.items():
            for comm in communities:
                if comm.id == community_id:
                    return comm.members
        return None

    @staticmethod
    def get_communities_for_entity(
        hierarchy: CommunityHierarchy,
        entity_id: str
    ) -> Dict[int, str]:
        """Get community assignments for an entity across all levels."""
        result = {}
        for level, mapping in hierarchy.entity_to_community.items():
            if entity_id in mapping:
                result[level] = mapping[entity_id]
        return result


class CommunityExporter:
    """
    Export communities to various formats.
    """

    @staticmethod
    def to_dict_list(hierarchy: CommunityHierarchy) -> List[Dict[str, Any]]:
        """Export communities as list of dictionaries."""
        result = []

        for level, communities in hierarchy.levels.items():
            for comm in communities:
                result.append({
                    "community_id": comm.id,
                    "level": comm.level,
                    "resolution": comm.resolution,
                    "member_count": comm.member_count,
                    "members": comm.members,
                    "internal_edges": comm.internal_edges,
                    "external_edges": comm.external_edges,
                    "parent_community": comm.parent_community,
                    "child_communities": comm.child_communities,
                })

        return result

    @staticmethod
    def to_dataframe(hierarchy: CommunityHierarchy):
        """Export communities as pandas DataFrame."""
        import pandas as pd

        data = CommunityExporter.to_dict_list(hierarchy)
        return pd.DataFrame(data)


# Export
__all__ = [
    'LeidenCommunityDetector',
    'CommunityAnalyzer',
    'CommunityExporter',
    'Community',
    'CommunityHierarchy',
    'CommunityConfig',
]
