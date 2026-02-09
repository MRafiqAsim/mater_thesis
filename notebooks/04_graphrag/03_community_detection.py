# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4.3: Community Detection
# MAGIC
# MAGIC Detect hierarchical communities in the knowledge graph using the Leiden algorithm.
# MAGIC
# MAGIC **Week 8 - Community Analysis**
# MAGIC
# MAGIC ## Features
# MAGIC - Multi-resolution Leiden community detection
# MAGIC - Hierarchical community structure (fine → coarse)
# MAGIC - Community statistics and quality metrics
# MAGIC - Parent-child relationships between levels
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install igraph leidenalg python-louvain delta-spark pandas matplotlib

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode
from pyspark.sql.types import ArrayType, StringType

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from graphrag.community_detection import (
    LeidenCommunityDetector,
    CommunityAnalyzer,
    CommunityExporter,
    CommunityConfig,
    CommunityHierarchy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure Delta Lake Paths
# Delta Lake paths
SILVER_PATH = "/mnt/datalake/silver"
GOLD_PATH = "/mnt/datalake/gold"

# Input: Graph data from Phase 4.2
ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"
GRAPH_EXPORT_TABLE = f"{GOLD_PATH}/graph_igraph_export"

# Output: Community detection results
COMMUNITIES_TABLE = f"{GOLD_PATH}/communities"
COMMUNITY_MEMBERS_TABLE = f"{GOLD_PATH}/community_members"
COMMUNITY_HIERARCHY_TABLE = f"{GOLD_PATH}/community_hierarchy"

print("Delta Lake paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Graph Data

# COMMAND ----------

# DBTITLE 1,Load Entities and Relationships
# Load entities
entities_df = spark.read.format("delta").load(ENTITIES_TABLE)
entities = [row.asDict() for row in entities_df.collect()]
print(f"Loaded {len(entities)} entities")

# Load relationships
relationships_df = spark.read.format("delta").load(RELATIONSHIPS_TABLE)
relationships = [row.asDict() for row in relationships_df.collect()]
print(f"Loaded {len(relationships)} relationships")

# COMMAND ----------

# DBTITLE 1,Prepare Vertices and Edges
# Prepare vertices for community detection
vertices = [
    {
        "id": e["id"],
        "name": e["name"],
        "type": e["type"]
    }
    for e in entities
]

# Prepare edges for community detection
edges = [
    {
        "source_id": r["source_id"],
        "target_id": r["target_id"],
        "strength": r.get("strength", 1.0),
        "weight": r.get("strength", 1.0)  # Alias for weight
    }
    for r in relationships
]

print(f"Prepared {len(vertices)} vertices and {len(edges)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Community Detection

# COMMAND ----------

# DBTITLE 1,Configure Leiden Algorithm
# Community detection configuration
community_config = CommunityConfig(
    # Multi-resolution parameters (fine to coarse)
    resolutions=[0.5, 1.0, 2.0],  # Level 0 (fine), Level 1 (medium), Level 2 (coarse)

    # Leiden algorithm parameters
    n_iterations=2,
    random_state=42,

    # Filtering
    min_community_size=3,  # Minimum entities per community
    max_communities_per_level=100  # Limit communities per level
)

print("Community detection configuration:")
print(f"  Resolutions: {community_config.resolutions}")
print(f"  Min community size: {community_config.min_community_size}")
print(f"  Max communities per level: {community_config.max_communities_per_level}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run Community Detection

# COMMAND ----------

# DBTITLE 1,Initialize Detector
# Initialize Leiden community detector
detector = LeidenCommunityDetector(config=community_config)

print("Leiden community detector initialized")

# COMMAND ----------

# DBTITLE 1,Detect Communities
print("Running community detection at multiple resolutions...")
print("-" * 60)

# Run detection
hierarchy = detector.detect(vertices=vertices, edges=edges)

print("-" * 60)
print(f"Community detection complete!")
print(f"  Total communities detected: {hierarchy.total_communities}")
print(f"  Hierarchy levels: {len(hierarchy.levels)}")

# COMMAND ----------

# DBTITLE 1,Display Community Statistics
# Get detailed statistics
stats = CommunityAnalyzer.get_community_statistics(hierarchy)

print("=" * 60)
print("COMMUNITY DETECTION STATISTICS")
print("=" * 60)

print(f"\n📊 OVERVIEW:")
print(f"  Total communities: {stats['total_communities']}")
print(f"  Number of levels: {stats['levels']}")

print(f"\n📈 MODULARITY SCORES:")
for level, score in stats['modularity_scores'].items():
    print(f"  Level {level}: {score:.4f}")

print(f"\n📁 BY LEVEL:")
for level, level_stats in stats['by_level'].items():
    print(f"\n  Level {level} (resolution={community_config.resolutions[level]}):")
    print(f"    Communities: {level_stats['num_communities']}")
    print(f"    Avg size: {level_stats['avg_size']:.1f}")
    print(f"    Min size: {level_stats['min_size']}")
    print(f"    Max size: {level_stats['max_size']}")
    print(f"    Total members: {level_stats['total_members']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Community Structure

# COMMAND ----------

# DBTITLE 1,Export Communities to DataFrame
# Export communities to list of dictionaries
communities_list = CommunityExporter.to_dict_list(hierarchy)

print(f"Exported {len(communities_list)} communities")

# Display sample
for comm in communities_list[:3]:
    print(f"\nCommunity: {comm['community_id']}")
    print(f"  Level: {comm['level']}")
    print(f"  Members: {comm['member_count']}")
    print(f"  Internal edges: {comm['internal_edges']}")
    print(f"  External edges: {comm['external_edges']}")
    print(f"  Parent: {comm['parent_community']}")
    print(f"  Children: {len(comm['child_communities'])}")

# COMMAND ----------

# DBTITLE 1,Visualize Community Size Distribution
import matplotlib.pyplot as plt

# Plot community size distribution by level
fig, axes = plt.subplots(1, len(hierarchy.levels), figsize=(15, 4))

for i, (level, communities) in enumerate(hierarchy.levels.items()):
    sizes = [c.member_count for c in communities]
    ax = axes[i] if len(hierarchy.levels) > 1 else axes

    ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(f'Level {level} (n={len(communities)})')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/tmp/community_size_distribution.png', dpi=150)
display()

# COMMAND ----------

# DBTITLE 1,Top Communities by Size
# Show largest communities at each level
print("=" * 60)
print("LARGEST COMMUNITIES BY LEVEL")
print("=" * 60)

for level, communities in hierarchy.levels.items():
    print(f"\nLevel {level}:")
    print("-" * 40)

    # Sort by size
    sorted_comms = sorted(communities, key=lambda c: c.member_count, reverse=True)

    for comm in sorted_comms[:5]:
        # Get member names
        member_names = []
        for member_id in comm.members[:5]:
            entity = next((e for e in entities if e["id"] == member_id), None)
            if entity:
                member_names.append(entity["name"])

        print(f"  {comm.id}: {comm.member_count} members")
        print(f"    Sample members: {', '.join(member_names)}")
        if len(comm.members) > 5:
            print(f"    ... and {len(comm.members) - 5} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyze Hierarchy Structure

# COMMAND ----------

# DBTITLE 1,Parent-Child Relationships
# Analyze hierarchy structure
print("=" * 60)
print("COMMUNITY HIERARCHY STRUCTURE")
print("=" * 60)

for level in sorted(hierarchy.levels.keys()):
    communities = hierarchy.levels[level]

    # Count parents and children
    has_parent = sum(1 for c in communities if c.parent_community)
    has_children = sum(1 for c in communities if c.child_communities)

    print(f"\nLevel {level}:")
    print(f"  Total communities: {len(communities)}")
    print(f"  With parent community: {has_parent}")
    print(f"  With child communities: {has_children}")

# COMMAND ----------

# DBTITLE 1,Entity Community Assignments
# Show how entities are assigned across levels
print("\nSample Entity Community Assignments:")
print("-" * 60)

sample_entities = entities[:5]
for entity in sample_entities:
    assignments = CommunityAnalyzer.get_communities_for_entity(hierarchy, entity["id"])
    print(f"\n{entity['name']} ({entity['type']}):")
    for level, comm_id in assignments.items():
        print(f"  Level {level}: {comm_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Results to Delta Lake

# COMMAND ----------

# DBTITLE 1,Save Communities
# Create communities DataFrame
communities_data = []
for comm in communities_list:
    communities_data.append({
        "community_id": comm["community_id"],
        "level": comm["level"],
        "resolution": comm["resolution"],
        "member_count": comm["member_count"],
        "internal_edges": comm["internal_edges"],
        "external_edges": comm["external_edges"],
        "parent_community": comm["parent_community"],
        "num_children": len(comm["child_communities"]),
        "created_at": datetime.now().isoformat()
    })

communities_df = spark.createDataFrame(communities_data)
communities_df.write.format("delta").mode("overwrite").save(COMMUNITIES_TABLE)
print(f"Saved {communities_df.count()} communities to {COMMUNITIES_TABLE}")

display(communities_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Save Community Members
# Create community members DataFrame (for join operations)
members_data = []
for comm in communities_list:
    for member_id in comm["members"]:
        members_data.append({
            "community_id": comm["community_id"],
            "level": comm["level"],
            "entity_id": member_id
        })

members_df = spark.createDataFrame(members_data)
members_df.write.format("delta").mode("overwrite").save(COMMUNITY_MEMBERS_TABLE)
print(f"Saved {members_df.count()} community-member mappings to {COMMUNITY_MEMBERS_TABLE}")

display(members_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Save Hierarchy Data
# Save hierarchy relationships
hierarchy_data = []
for comm in communities_list:
    if comm["child_communities"]:
        for child_id in comm["child_communities"]:
            hierarchy_data.append({
                "parent_community": comm["community_id"],
                "parent_level": comm["level"],
                "child_community": child_id,
                "child_level": comm["level"] - 1 if comm["level"] > 0 else 0
            })

if hierarchy_data:
    hierarchy_df = spark.createDataFrame(hierarchy_data)
    hierarchy_df.write.format("delta").mode("overwrite").save(COMMUNITY_HIERARCHY_TABLE)
    print(f"Saved {hierarchy_df.count()} hierarchy relationships to {COMMUNITY_HIERARCHY_TABLE}")
else:
    print("No hierarchy relationships to save")

# COMMAND ----------

# DBTITLE 1,Save Full Community Data for Summarization
# Save full community data including member lists (for summarization)
full_communities_data = []
for comm in communities_list:
    full_communities_data.append({
        "community_id": comm["community_id"],
        "level": comm["level"],
        "resolution": comm["resolution"],
        "member_count": comm["member_count"],
        "members": json.dumps(comm["members"]),  # Store as JSON string
        "internal_edges": comm["internal_edges"],
        "external_edges": comm["external_edges"],
        "parent_community": comm["parent_community"],
        "child_communities": json.dumps(comm["child_communities"])
    })

full_communities_df = spark.createDataFrame(full_communities_data)
full_communities_df.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/communities_full")
print(f"Saved full community data to {GOLD_PATH}/communities_full")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Community Quality Metrics

# COMMAND ----------

# DBTITLE 1,Calculate Quality Metrics
import igraph as ig
import leidenalg

# Rebuild igraph for quality analysis
id_to_idx = {v["id"]: i for i, v in enumerate(vertices)}

g = ig.Graph(directed=False)
g.add_vertices(len(vertices))
g.vs["entity_id"] = [v["id"] for v in vertices]
g.vs["name"] = [v["name"] for v in vertices]

edge_list = []
weights = []
for edge in edges:
    source_idx = id_to_idx.get(edge["source_id"])
    target_idx = id_to_idx.get(edge["target_id"])
    if source_idx is not None and target_idx is not None:
        edge_list.append((source_idx, target_idx))
        weights.append(edge.get("weight", 1.0))

if edge_list:
    g.add_edges(edge_list)
    g.es["weight"] = weights

print("Quality Metrics by Level:")
print("-" * 60)

for level, communities in hierarchy.levels.items():
    # Create membership list
    membership = [-1] * g.vcount()
    for i, comm in enumerate(communities):
        for member_id in comm.members:
            if member_id in id_to_idx:
                membership[id_to_idx[member_id]] = i

    # Calculate metrics
    if g.ecount() > 0:
        # Coverage: fraction of intra-community edges
        intra_edges = sum(c.internal_edges for c in communities)
        total_edges = g.ecount()
        coverage = intra_edges / total_edges if total_edges > 0 else 0

        # Modularity
        partition = leidenalg.RBConfigurationVertexPartition(
            g,
            initial_membership=membership,
            resolution_parameter=community_config.resolutions[level]
        )
        modularity = partition.modularity

        print(f"\nLevel {level}:")
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Communities: {len(communities)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Community Detection Summary
print("=" * 60)
print("PHASE 4.3: COMMUNITY DETECTION COMPLETE")
print("=" * 60)

print(f"\n📊 DETECTION RESULTS:")
print(f"  Total communities: {hierarchy.total_communities}")
print(f"  Hierarchy levels: {len(hierarchy.levels)}")
print(f"  Resolutions used: {community_config.resolutions}")

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"  Communities: {COMMUNITIES_TABLE}")
print(f"  Community members: {COMMUNITY_MEMBERS_TABLE}")
print(f"  Hierarchy: {COMMUNITY_HIERARCHY_TABLE}")
print(f"  Full data: {GOLD_PATH}/communities_full")

print(f"\n📈 BY LEVEL:")
for level, communities in hierarchy.levels.items():
    modularity = hierarchy.modularity_scores.get(level, 0)
    print(f"  Level {level}: {len(communities)} communities (modularity: {modularity:.4f})")

print("\n" + "=" * 60)
print("NEXT: 04_community_summarization.py - Generate Summaries")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Leiden Algorithm:**
# MAGIC - Multi-resolution community detection
# MAGIC - Higher resolution = smaller, more granular communities
# MAGIC - Lower resolution = larger, more general communities
# MAGIC
# MAGIC **Hierarchy Structure:**
# MAGIC - Level 0 (fine): Small, specific topic clusters
# MAGIC - Level 1 (medium): Moderate-sized thematic groups
# MAGIC - Level 2 (coarse): Large, general categories
# MAGIC
# MAGIC **Quality Metrics:**
# MAGIC - Modularity: Measures quality of community structure
# MAGIC - Coverage: Fraction of edges within communities
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Run `04_community_summarization.py` to generate GPT-4o summaries
# MAGIC 2. Index community summaries for retrieval
# MAGIC 3. Integrate into GraphRAG retrieval pipeline
