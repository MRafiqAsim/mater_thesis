# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4.2: Knowledge Graph Construction
# MAGIC
# MAGIC Build and populate Azure Cosmos DB Gremlin knowledge graph with extracted entities and relationships.
# MAGIC
# MAGIC **Week 7 - Graph Database Integration**
# MAGIC
# MAGIC ## Features
# MAGIC - Cosmos DB Gremlin API integration
# MAGIC - Entity (vertex) population with properties
# MAGIC - Relationship (edge) population with weights
# MAGIC - Graph traversal queries
# MAGIC - Graph statistics and validation
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install gremlinpython tenacity delta-spark networkx

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
from pyspark.sql.functions import col, lit

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from graphrag.graph_store import (
    CosmosGraphStore,
    InMemoryGraphStore,
    GraphConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
# Load Cosmos DB credentials from Databricks secrets
COSMOS_GREMLIN_ENDPOINT = dbutils.secrets.get(scope="azure-cosmos", key="gremlin-endpoint")
COSMOS_KEY = dbutils.secrets.get(scope="azure-cosmos", key="primary-key")
COSMOS_DATABASE = dbutils.secrets.get(scope="azure-cosmos", key="database")
COSMOS_GRAPH = dbutils.secrets.get(scope="azure-cosmos", key="graph")

# For development/testing, use in-memory store
USE_IN_MEMORY = True  # Set to False for production Cosmos DB

print(f"Cosmos DB Database: {COSMOS_DATABASE}")
print(f"Cosmos DB Graph: {COSMOS_GRAPH}")
print(f"Using in-memory store: {USE_IN_MEMORY}")

# COMMAND ----------

# DBTITLE 1,Configure Delta Lake Paths
# Delta Lake paths
SILVER_PATH = "/mnt/datalake/silver"
GOLD_PATH = "/mnt/datalake/gold"

# Input: Extracted entities and relationships from Phase 4.1
ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"

# Output: Graph statistics
GRAPH_STATS_TABLE = f"{GOLD_PATH}/graph_statistics"

print("Delta Lake paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Graph Store

# COMMAND ----------

# DBTITLE 1,Initialize Graph Store
if USE_IN_MEMORY:
    # Use in-memory store for development
    graph_store = InMemoryGraphStore()
    print("Initialized in-memory graph store")
else:
    # Use Cosmos DB Gremlin for production
    graph_config = GraphConfig(
        endpoint=COSMOS_GREMLIN_ENDPOINT,
        database=COSMOS_DATABASE,
        graph=COSMOS_GRAPH,
        partition_key="/type",
        batch_size=100
    )

    graph_store = CosmosGraphStore(
        endpoint=COSMOS_GREMLIN_ENDPOINT,
        key=COSMOS_KEY,
        config=graph_config
    )
    print(f"Connected to Cosmos DB Gremlin: {COSMOS_DATABASE}/{COSMOS_GRAPH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Entities and Relationships

# COMMAND ----------

# DBTITLE 1,Load Entities from Delta Lake
# Load normalized entities
entities_df = spark.read.format("delta").load(ENTITIES_TABLE)

print(f"Loaded {entities_df.count()} entities")
entities_df.printSchema()

# Display sample
display(entities_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Load Relationships from Delta Lake
# Load relationships
relationships_df = spark.read.format("delta").load(RELATIONSHIPS_TABLE)

print(f"Loaded {relationships_df.count()} relationships")
relationships_df.printSchema()

# Display sample
display(relationships_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Convert to Python Lists
# Convert to Python dictionaries for graph population
entities = [row.asDict() for row in entities_df.collect()]
relationships = [row.asDict() for row in relationships_df.collect()]

print(f"Prepared {len(entities)} entities and {len(relationships)} relationships for graph population")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Populate Knowledge Graph

# COMMAND ----------

# DBTITLE 1,Add Entities (Vertices)
from tqdm import tqdm

print(f"Adding {len(entities)} entities to graph...")
print("-" * 60)

success_count = 0
error_count = 0

for entity in tqdm(entities, desc="Adding entities"):
    try:
        # Prepare entity for graph store
        entity_data = {
            "id": entity["id"],
            "name": entity["name"],
            "type": entity["type"],
            "description": entity.get("description", ""),
            "mention_count": entity.get("mention_count", 1),
            "properties": {
                "chunk_count": entity.get("chunk_count", 0),
                "source_count": entity.get("source_count", 0),
            }
        }

        graph_store.add_entity(entity_data)
        success_count += 1

    except Exception as e:
        logger.warning(f"Failed to add entity {entity['id']}: {e}")
        error_count += 1

print("-" * 60)
print(f"Entities added: {success_count} success, {error_count} errors")

# COMMAND ----------

# DBTITLE 1,Add Relationships (Edges)
print(f"Adding {len(relationships)} relationships to graph...")
print("-" * 60)

rel_success = 0
rel_errors = 0

for rel in tqdm(relationships, desc="Adding relationships"):
    try:
        # Prepare relationship for graph store
        rel_data = {
            "id": rel["id"],
            "source_id": rel["source_id"],
            "target_id": rel["target_id"],
            "type": rel["type"],
            "description": rel.get("description", ""),
            "strength": rel.get("strength", 1.0),
        }

        graph_store.add_relationship(rel_data)
        rel_success += 1

    except Exception as e:
        logger.warning(f"Failed to add relationship {rel['id']}: {e}")
        rel_errors += 1

print("-" * 60)
print(f"Relationships added: {rel_success} success, {rel_errors} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Graph Validation and Statistics

# COMMAND ----------

# DBTITLE 1,Get Graph Statistics
# Get graph statistics
stats = graph_store.get_statistics()

print("=" * 60)
print("KNOWLEDGE GRAPH STATISTICS")
print("=" * 60)

print(f"\n📊 OVERVIEW:")
print(f"  Total vertices (entities): {stats['vertex_count']}")
print(f"  Total edges (relationships): {stats['edge_count']}")

print(f"\n📁 VERTICES BY TYPE:")
for entity_type, count in stats.get('vertices_by_type', {}).items():
    print(f"  {entity_type}: {count}")

print(f"\n🔗 EDGES BY TYPE:")
for edge_type, count in stats.get('edges_by_type', {}).items():
    print(f"  {edge_type}: {count}")

# COMMAND ----------

# DBTITLE 1,Validate Graph Integrity
# Validate graph integrity
print("Validating graph integrity...")

# Check for orphan relationships (edges with missing vertices)
if USE_IN_MEMORY:
    vertex_ids = set(graph_store.vertices.keys())

    orphan_edges = 0
    for edge in graph_store.edges:
        if edge["source_id"] not in vertex_ids or edge["target_id"] not in vertex_ids:
            orphan_edges += 1

    print(f"  Orphan edges (missing vertices): {orphan_edges}")

    # Calculate graph density
    n = len(vertex_ids)
    m = len(graph_store.edges)
    if n > 1:
        density = (2 * m) / (n * (n - 1))
        print(f"  Graph density: {density:.6f}")

    # Calculate average degree
    degree_counts = {}
    for edge in graph_store.edges:
        degree_counts[edge["source_id"]] = degree_counts.get(edge["source_id"], 0) + 1
        degree_counts[edge["target_id"]] = degree_counts.get(edge["target_id"], 0) + 1

    if degree_counts:
        avg_degree = sum(degree_counts.values()) / len(degree_counts)
        max_degree = max(degree_counts.values())
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Max degree: {max_degree}")

print("\n✅ Graph validation complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Graph Traversal Examples

# COMMAND ----------

# DBTITLE 1,Export to NetworkX for Analysis
if USE_IN_MEMORY:
    import networkx as nx

    # Export to NetworkX
    G = graph_store.to_networkx()

    print(f"NetworkX graph created:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Is connected: {nx.is_weakly_connected(G)}")

    # Find connected components
    components = list(nx.weakly_connected_components(G))
    print(f"  Connected components: {len(components)}")

    # Largest component size
    if components:
        largest = max(components, key=len)
        print(f"  Largest component size: {len(largest)}")

# COMMAND ----------

# DBTITLE 1,Find Most Connected Entities
if USE_IN_MEMORY:
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)

    # Get top 10 most connected entities
    top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 Most Connected Entities:")
    print("-" * 60)
    for entity_id, centrality in top_connected:
        entity = graph_store.vertices.get(entity_id, {})
        name = entity.get("name", entity_id)
        entity_type = entity.get("type", "UNKNOWN")
        degree = G.degree(entity_id)
        print(f"  {name} ({entity_type}): degree={degree}, centrality={centrality:.4f}")

# COMMAND ----------

# DBTITLE 1,Sample Graph Queries
if USE_IN_MEMORY:
    # Example: Find all entities connected to a specific entity
    sample_entity_id = list(graph_store.vertices.keys())[0] if graph_store.vertices else None

    if sample_entity_id:
        sample_entity = graph_store.vertices[sample_entity_id]
        print(f"\nNeighbors of '{sample_entity.get('name', sample_entity_id)}':")
        print("-" * 40)

        # Find neighbors
        neighbors = list(G.neighbors(sample_entity_id))
        for neighbor_id in neighbors[:10]:  # Limit to 10
            neighbor = graph_store.vertices.get(neighbor_id, {})
            print(f"  → {neighbor.get('name', neighbor_id)} ({neighbor.get('type', 'UNKNOWN')})")

        if len(neighbors) > 10:
            print(f"  ... and {len(neighbors) - 10} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Export for Community Detection

# COMMAND ----------

# DBTITLE 1,Export to igraph Format
if USE_IN_MEMORY:
    # Export to igraph for community detection
    ig_graph = graph_store.to_igraph()

    print(f"igraph export created:")
    print(f"  Vertices: {ig_graph.vcount()}")
    print(f"  Edges: {ig_graph.ecount()}")
    print(f"  Is connected: {ig_graph.is_connected()}")

    # Save igraph data for next notebook
    igraph_data = {
        "vertices": [
            {
                "id": ig_graph.vs[i]["id"],
                "name": ig_graph.vs[i]["name"],
                "type": ig_graph.vs[i]["type"]
            }
            for i in range(ig_graph.vcount())
        ],
        "edges": [
            {
                "source_id": ig_graph.vs[ig_graph.es[i].source]["id"],
                "target_id": ig_graph.vs[ig_graph.es[i].target]["id"],
                "weight": ig_graph.es[i]["weight"]
            }
            for i in range(ig_graph.ecount())
        ]
    }

    # Save to JSON for community detection
    import json
    igraph_json = json.dumps(igraph_data)

    # Save as Delta table
    igraph_df = spark.createDataFrame([{"graph_data": igraph_json}])
    igraph_df.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/graph_igraph_export")
    print(f"\nSaved igraph export to {GOLD_PATH}/graph_igraph_export")

# COMMAND ----------

# DBTITLE 1,Save Graph Statistics
# Save statistics to Delta Lake
stats_record = {
    "timestamp": datetime.now().isoformat(),
    "vertex_count": stats["vertex_count"],
    "edge_count": stats["edge_count"],
    "vertices_by_type": json.dumps(stats.get("vertices_by_type", {})),
    "edges_by_type": json.dumps(stats.get("edges_by_type", {})),
    "storage_type": "in_memory" if USE_IN_MEMORY else "cosmos_db"
}

stats_df = spark.createDataFrame([stats_record])
stats_df.write.format("delta").mode("append").save(GRAPH_STATS_TABLE)
print(f"Saved graph statistics to {GRAPH_STATS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Knowledge Graph Summary
print("=" * 60)
print("PHASE 4.2: KNOWLEDGE GRAPH CONSTRUCTION COMPLETE")
print("=" * 60)

print(f"\n📊 GRAPH STATISTICS:")
print(f"  Vertices (entities): {stats['vertex_count']}")
print(f"  Edges (relationships): {stats['edge_count']}")
print(f"  Storage: {'In-Memory' if USE_IN_MEMORY else 'Cosmos DB Gremlin'}")

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"  Graph statistics: {GRAPH_STATS_TABLE}")
print(f"  igraph export: {GOLD_PATH}/graph_igraph_export")

print(f"\n🎯 VERTEX BREAKDOWN:")
for entity_type, count in stats.get('vertices_by_type', {}).items():
    print(f"  {entity_type}: {count}")

print(f"\n🔗 EDGE BREAKDOWN:")
for edge_type, count in stats.get('edges_by_type', {}).items():
    print(f"  {edge_type}: {count}")

print("\n" + "=" * 60)
print("NEXT: 03_community_detection.py - Detect Communities")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Graph Storage Options:**
# MAGIC 1. **In-Memory (Development)**: Fast iteration, exports to NetworkX/igraph
# MAGIC 2. **Cosmos DB Gremlin (Production)**: Scalable, persistent, supports Gremlin queries
# MAGIC
# MAGIC **Key Graph Metrics:**
# MAGIC - Density: Ratio of actual edges to possible edges
# MAGIC - Average Degree: Average number of connections per entity
# MAGIC - Connected Components: Isolated subgraphs
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Run `03_community_detection.py` to detect communities using Leiden algorithm
# MAGIC 2. Run `04_community_summarization.py` to generate community summaries
# MAGIC 3. Integrate summaries into RAG retrieval pipeline
