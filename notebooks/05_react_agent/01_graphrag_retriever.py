# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 5.1: GraphRAG Retriever
# MAGIC
# MAGIC Combined retrieval integrating vector search, graph traversal, and community summaries.
# MAGIC
# MAGIC **Week 9 - Retrieval Pipeline Integration**
# MAGIC
# MAGIC ## Features
# MAGIC - Hybrid vector + keyword search
# MAGIC - Knowledge graph entity lookup
# MAGIC - Community summary retrieval
# MAGIC - Query type classification (local vs global)
# MAGIC - Multi-source context aggregation
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai azure-search-documents numpy delta-spark

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from agents.graphrag_retriever import (
    GraphRAGRetriever,
    GraphRAGConfig,
    GraphRAGContext,
    QueryType,
    QueryClassifier
)
from graphrag.graph_store import InMemoryGraphStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
# Load credentials from Databricks secrets
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="azure-openai", key="endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="azure-openai", key="api-key")
AZURE_SEARCH_ENDPOINT = dbutils.secrets.get(scope="azure-search", key="endpoint")
AZURE_SEARCH_KEY = dbutils.secrets.get(scope="azure-search", key="admin-key")

print("Azure credentials loaded")

# COMMAND ----------

# DBTITLE 1,Configure Delta Lake Paths
GOLD_PATH = "/mnt/datalake/gold"

# Input tables
ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"
COMMUNITY_SUMMARIES_TABLE = f"{GOLD_PATH}/community_summaries_indexed"
CHUNKS_TABLE = f"{GOLD_PATH}/chunks_embedded"

print("Delta Lake paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Knowledge Graph Data

# COMMAND ----------

# DBTITLE 1,Load Entities
entities_df = spark.read.format("delta").load(ENTITIES_TABLE)
entities = [row.asDict() for row in entities_df.collect()]

print(f"Loaded {len(entities)} entities")
display(entities_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Load Relationships
relationships_df = spark.read.format("delta").load(RELATIONSHIPS_TABLE)
relationships = [row.asDict() for row in relationships_df.collect()]

print(f"Loaded {len(relationships)} relationships")
display(relationships_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Load Community Summaries with Embeddings
summaries_df = spark.read.format("delta").load(COMMUNITY_SUMMARIES_TABLE)
summaries_raw = summaries_df.collect()

# Parse embeddings from JSON
community_summaries = []
community_embeddings = []

for row in summaries_raw:
    summary = row.asDict()

    # Parse JSON fields
    if "key_entities" in summary and isinstance(summary["key_entities"], str):
        summary["key_entities"] = json.loads(summary["key_entities"])
    if "key_themes" in summary and isinstance(summary["key_themes"], str):
        summary["key_themes"] = json.loads(summary["key_themes"])

    # Extract embedding
    embedding = json.loads(summary.pop("embedding", "[]"))

    community_summaries.append(summary)
    community_embeddings.append(embedding)

community_embeddings = np.array(community_embeddings)

print(f"Loaded {len(community_summaries)} community summaries")
print(f"Embedding shape: {community_embeddings.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Vector Store

# COMMAND ----------

# DBTITLE 1,Initialize Azure AI Search
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-01"
)

# Initialize Azure Search vector store
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name="document-chunks",  # From Phase 3
    embedding_function=embeddings.embed_query,
)

print("Vector store initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize Graph Store

# COMMAND ----------

# DBTITLE 1,Build In-Memory Graph Store
# Initialize in-memory graph store
graph_store = InMemoryGraphStore()

# Add entities
for entity in entities:
    graph_store.add_entity(entity)

# Add relationships
for rel in relationships:
    graph_store.add_relationship(rel)

stats = graph_store.get_statistics()
print(f"Graph store statistics:")
print(f"  Vertices: {stats['vertex_count']}")
print(f"  Edges: {stats['edge_count']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure GraphRAG Retriever

# COMMAND ----------

# DBTITLE 1,Configure Retriever
# GraphRAG configuration
retriever_config = GraphRAGConfig(
    # Vector search
    vector_top_k=10,
    vector_score_threshold=0.7,

    # Graph search
    entity_top_k=10,
    max_graph_hops=2,
    relationship_limit=20,

    # Community search
    community_top_k=5,
    prefer_level=1,  # Medium granularity

    # Context limits
    max_context_tokens=8000,

    # Query routing keywords
    local_keywords=["who", "what", "when", "where", "which", "name", "specific", "exactly"],
    global_keywords=["overview", "summary", "general", "trends", "themes", "overall", "main", "describe"]
)

print("Retriever configuration:")
print(f"  Vector top-k: {retriever_config.vector_top_k}")
print(f"  Entity top-k: {retriever_config.entity_top_k}")
print(f"  Community top-k: {retriever_config.community_top_k}")

# COMMAND ----------

# DBTITLE 1,Initialize GraphRAG Retriever
# Initialize retriever
retriever = GraphRAGRetriever(
    vector_store=vector_store,
    graph_store=graph_store,
    community_summaries=community_summaries,
    community_embeddings=community_embeddings,
    embeddings=embeddings,
    entities=entities,
    relationships=relationships,
    config=retriever_config
)

print("GraphRAG retriever initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Query Classification

# COMMAND ----------

# DBTITLE 1,Test Query Types
# Test query classifier
classifier = QueryClassifier(retriever_config)

test_queries = [
    "Who is John Smith?",
    "What projects is the engineering team working on?",
    "Give me an overview of the company's technology stack",
    "What are the main themes in recent communications?",
    "When did Project Alpha start?",
    "Describe the organizational structure",
    "What is the relationship between Sales and Marketing?",
]

print("Query Classification Results:")
print("-" * 60)

for query in test_queries:
    query_type = classifier.classify(query)
    print(f"  '{query[:50]}...' => {query_type.value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Retrieval

# COMMAND ----------

# DBTITLE 1,Test Local Query
# Test local (specific) query
local_query = "What projects is John working on?"

print(f"Query: {local_query}")
print(f"Query Type: {classifier.classify(local_query).value}")
print("-" * 60)

context = retriever.retrieve(local_query)

print(f"\n📊 RETRIEVAL RESULTS:")
print(f"  Query type: {context.query_type.value}")
print(f"  Chunks found: {len(context.chunks)}")
print(f"  Entities found: {len(context.entities)}")
print(f"  Communities found: {len(context.communities)}")
print(f"  Relationships found: {len(context.relationships)}")

# Show top results
if context.entities:
    print(f"\n👤 TOP ENTITIES:")
    for r in context.entities[:3]:
        print(f"  - {r.content[:100]}...")

if context.relationships:
    print(f"\n🔗 TOP RELATIONSHIPS:")
    for r in context.relationships[:3]:
        print(f"  - {r.content}")

if context.chunks:
    print(f"\n📄 TOP CHUNKS:")
    for r in context.chunks[:2]:
        print(f"  - [{r.metadata.get('source_file', 'Unknown')}] {r.content[:100]}...")

# COMMAND ----------

# DBTITLE 1,Test Global Query
# Test global (summary) query
global_query = "What are the main themes and projects across the organization?"

print(f"Query: {global_query}")
print(f"Query Type: {classifier.classify(global_query).value}")
print("-" * 60)

context = retriever.retrieve(global_query)

print(f"\n📊 RETRIEVAL RESULTS:")
print(f"  Query type: {context.query_type.value}")
print(f"  Chunks found: {len(context.chunks)}")
print(f"  Entities found: {len(context.entities)}")
print(f"  Communities found: {len(context.communities)}")
print(f"  Relationships found: {len(context.relationships)}")

# Show community summaries (primary for global queries)
if context.communities:
    print(f"\n🏘️ TOP COMMUNITIES:")
    for r in context.communities[:3]:
        print(f"\n  Community {r.source_id} (Level {r.metadata.get('level', '?')}):")
        print(f"    {r.content[:200]}...")
        themes = r.metadata.get('key_themes', [])
        if themes:
            print(f"    Themes: {', '.join(themes[:3])}")

# COMMAND ----------

# DBTITLE 1,Test Entity-Focused Retrieval
# Test entity-focused retrieval
if entities:
    sample_entity_id = entities[0]["id"]
    sample_entity_name = entities[0]["name"]

    print(f"Entity-focused retrieval for: {sample_entity_name}")
    print("-" * 60)

    context = retriever.retrieve_for_entity(sample_entity_id, include_neighbors=True)

    print(f"\n📊 RETRIEVAL RESULTS:")
    print(f"  Entities (including neighbors): {len(context.entities)}")
    print(f"  Relationships: {len(context.relationships)}")
    print(f"  Related chunks: {len(context.chunks)}")
    print(f"  Related communities: {len(context.communities)}")

    # Show entity neighborhood
    if context.entities:
        print(f"\n👥 ENTITY NEIGHBORHOOD:")
        for r in context.entities[:5]:
            print(f"  - {r.content[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Format Context for LLM

# COMMAND ----------

# DBTITLE 1,Format Context
# Test context formatting
test_query = "What is the relationship between the engineering team and recent projects?"

context = retriever.retrieve(test_query)

# Format for LLM
formatted_context = context.format_context(max_tokens=4000)

print(f"Query: {test_query}")
print(f"Query Type: {context.query_type.value}")
print("-" * 60)
print(f"\nFormatted Context ({len(formatted_context)} chars):\n")
print(formatted_context[:2000])
if len(formatted_context) > 2000:
    print(f"\n... [truncated, {len(formatted_context) - 2000} more characters]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Retrieval Performance Analysis

# COMMAND ----------

# DBTITLE 1,Analyze Retrieval Distribution
# Analyze retrieval across different query types
test_queries = [
    ("Who is the CEO?", QueryType.LOCAL),
    ("What projects use Python?", QueryType.LOCAL),
    ("Summarize the company's main activities", QueryType.GLOBAL),
    ("What are the key themes in communications?", QueryType.GLOBAL),
    ("How does the sales team interact with engineering?", QueryType.HYBRID),
]

results_summary = []

for query, expected_type in test_queries:
    context = retriever.retrieve(query)

    results_summary.append({
        "query": query[:50] + "...",
        "expected_type": expected_type.value,
        "actual_type": context.query_type.value,
        "chunks": len(context.chunks),
        "entities": len(context.entities),
        "communities": len(context.communities),
        "relationships": len(context.relationships),
        "total_results": len(context.get_all_results())
    })

# Display summary
results_df = spark.createDataFrame(results_summary)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Retriever Configuration

# COMMAND ----------

# DBTITLE 1,Save Configuration
# Save retriever configuration for use in agent notebook
config_data = {
    "vector_top_k": retriever_config.vector_top_k,
    "vector_score_threshold": retriever_config.vector_score_threshold,
    "entity_top_k": retriever_config.entity_top_k,
    "max_graph_hops": retriever_config.max_graph_hops,
    "relationship_limit": retriever_config.relationship_limit,
    "community_top_k": retriever_config.community_top_k,
    "prefer_level": retriever_config.prefer_level,
    "max_context_tokens": retriever_config.max_context_tokens,
    "created_at": datetime.now().isoformat()
}

config_df = spark.createDataFrame([config_data])
config_df.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/retriever_config")

print(f"Saved retriever configuration to {GOLD_PATH}/retriever_config")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

# DBTITLE 1,Phase 5.1 Summary
print("=" * 60)
print("PHASE 5.1: GRAPHRAG RETRIEVER COMPLETE")
print("=" * 60)

print(f"\n📊 DATA SOURCES:")
print(f"  Entities: {len(entities)}")
print(f"  Relationships: {len(relationships)}")
print(f"  Community summaries: {len(community_summaries)}")
print(f"  Vector store: Azure AI Search")

print(f"\n🔧 RETRIEVER CAPABILITIES:")
print(f"  - Query classification (local/global/hybrid)")
print(f"  - Vector similarity search")
print(f"  - Entity and relationship lookup")
print(f"  - Community summary retrieval")
print(f"  - Entity-focused neighborhood retrieval")
print(f"  - Context formatting for LLM")

print(f"\n📁 OUTPUT:")
print(f"  Configuration: {GOLD_PATH}/retriever_config")

print("\n" + "=" * 60)
print("NEXT: 02_react_agent.py - Build ReAct Agent")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Query Routing:**
# MAGIC - LOCAL queries → Focus on entities, relationships, specific chunks
# MAGIC - GLOBAL queries → Focus on community summaries, themes
# MAGIC - HYBRID queries → Balanced retrieval from all sources
# MAGIC
# MAGIC **Context Aggregation:**
# MAGIC 1. Community summaries (high-level themes)
# MAGIC 2. Entity information (key facts)
# MAGIC 3. Relationships (connections)
# MAGIC 4. Document chunks (detailed evidence)
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Build ReAct agent with tool-augmented reasoning
# MAGIC 2. Test multi-hop question answering
# MAGIC 3. Evaluate against baseline RAG
