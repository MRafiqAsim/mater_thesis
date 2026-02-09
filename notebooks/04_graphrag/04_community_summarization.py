# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4.4: Community Summarization
# MAGIC
# MAGIC Generate LLM summaries for detected communities using GPT-4o.
# MAGIC
# MAGIC **Week 8 - Community Summaries for GraphRAG**
# MAGIC
# MAGIC ## Features
# MAGIC - GPT-4o community summarization
# MAGIC - Key entity and theme extraction
# MAGIC - Hierarchical summary structure
# MAGIC - Summary indexing for retrieval
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai tenacity delta-spark azure-search-documents

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
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import StringType, ArrayType

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from graphrag.community_summarization import (
    CommunitySummarizer,
    HierarchicalSummarizer,
    CommunitySummaryIndexer,
    CommunitySummary,
    SummarizationConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
# Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="azure-openai", key="endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="azure-openai", key="api-key")

# Azure AI Search credentials (for indexing)
AZURE_SEARCH_ENDPOINT = dbutils.secrets.get(scope="azure-search", key="endpoint")
AZURE_SEARCH_KEY = dbutils.secrets.get(scope="azure-search", key="admin-key")

print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT[:50]}...")
print(f"Azure Search Endpoint: {AZURE_SEARCH_ENDPOINT[:50]}...")

# COMMAND ----------

# DBTITLE 1,Configure Delta Lake Paths
# Delta Lake paths
GOLD_PATH = "/mnt/datalake/gold"

# Input: Communities and entities from previous steps
COMMUNITIES_TABLE = f"{GOLD_PATH}/communities_full"
ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"

# Output: Community summaries
SUMMARIES_TABLE = f"{GOLD_PATH}/community_summaries"
SUMMARIES_INDEXED_TABLE = f"{GOLD_PATH}/community_summaries_indexed"

print("Delta Lake paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Summarizer

# COMMAND ----------

# DBTITLE 1,Configure Summarization
# Summarization configuration
summarization_config = SummarizationConfig(
    model_deployment="gpt-4o",
    temperature=0.3,  # Some creativity for natural summaries
    max_tokens=800,
    # Content settings
    max_entities_in_prompt=30,
    max_relationships_in_prompt=50,
    max_context_length=10000
)

print("Summarization configuration:")
print(f"  Model: {summarization_config.model_deployment}")
print(f"  Temperature: {summarization_config.temperature}")
print(f"  Max tokens: {summarization_config.max_tokens}")

# COMMAND ----------

# DBTITLE 1,Initialize Summarizer
# Initialize community summarizer
summarizer = CommunitySummarizer(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    config=summarization_config
)

print("Community summarizer initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Communities and Entities

# COMMAND ----------

# DBTITLE 1,Load Communities
# Load full community data
communities_df = spark.read.format("delta").load(COMMUNITIES_TABLE)
communities_raw = communities_df.collect()

# Parse JSON fields
communities = []
for row in communities_raw:
    comm = row.asDict()
    comm["members"] = json.loads(comm["members"])
    comm["child_communities"] = json.loads(comm["child_communities"]) if comm.get("child_communities") else []
    communities.append(comm)

print(f"Loaded {len(communities)} communities")

# Group by level
communities_by_level = {}
for comm in communities:
    level = comm["level"]
    if level not in communities_by_level:
        communities_by_level[level] = []
    communities_by_level[level].append(comm)

print(f"Communities by level:")
for level, comms in sorted(communities_by_level.items()):
    print(f"  Level {level}: {len(comms)} communities")

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

# MAGIC %md
# MAGIC ## 4. Generate Community Summaries

# COMMAND ----------

# DBTITLE 1,Define Progress Callback
def progress_callback(current: int, total: int):
    """Progress callback for batch summarization."""
    if current % 5 == 0 or current == total:
        print(f"  Progress: {current}/{total} ({100*current/total:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Summarize Communities by Level
from tqdm import tqdm

all_summaries = []
total_summaries = 0
errors = 0

print("=" * 60)
print("GENERATING COMMUNITY SUMMARIES")
print("=" * 60)

# Process each level (fine to coarse)
for level in sorted(communities_by_level.keys()):
    level_communities = communities_by_level[level]

    print(f"\n📊 Level {level}: {len(level_communities)} communities")
    print("-" * 40)

    level_summaries = []

    for comm in tqdm(level_communities, desc=f"Level {level}"):
        try:
            # Generate summary
            summary = summarizer.summarize_community(
                community_id=comm["community_id"],
                level=comm["level"],
                member_ids=comm["members"],
                entities=entities,
                relationships=relationships,
                chunk_ids=[]  # Could track source chunks
            )

            level_summaries.append(summary)
            total_summaries += 1

        except Exception as e:
            logger.warning(f"Failed to summarize {comm['community_id']}: {e}")
            errors += 1

            # Create placeholder summary
            level_summaries.append(CommunitySummary(
                community_id=comm["community_id"],
                level=comm["level"],
                summary=f"Community of {comm['member_count']} entities",
                key_entities=[],
                key_themes=[],
                member_count=comm["member_count"],
                source_chunks=[]
            ))

    all_summaries.extend(level_summaries)

    # Show sample summaries for this level
    print(f"\n  Sample summaries for Level {level}:")
    for summary in level_summaries[:2]:
        print(f"\n  {summary.community_id}:")
        print(f"    Summary: {summary.summary[:200]}...")
        print(f"    Key entities: {', '.join(summary.key_entities[:3])}")
        print(f"    Key themes: {', '.join(summary.key_themes[:3])}")

print("\n" + "=" * 60)
print(f"Summarization complete: {total_summaries} summaries, {errors} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Summaries

# COMMAND ----------

# DBTITLE 1,Summary Statistics
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Calculate statistics
summary_lengths = [len(s.summary) for s in all_summaries]
avg_length = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0

theme_counts = [len(s.key_themes) for s in all_summaries]
avg_themes = sum(theme_counts) / len(theme_counts) if theme_counts else 0

entity_counts = [len(s.key_entities) for s in all_summaries]
avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0

print(f"\n📝 SUMMARY LENGTHS:")
print(f"  Average: {avg_length:.0f} characters")
print(f"  Min: {min(summary_lengths)} characters")
print(f"  Max: {max(summary_lengths)} characters")

print(f"\n🏷️ KEY THEMES:")
print(f"  Average per community: {avg_themes:.1f}")
print(f"  Total unique themes: {len(set(t for s in all_summaries for t in s.key_themes))}")

print(f"\n👥 KEY ENTITIES:")
print(f"  Average per community: {avg_entities:.1f}")

# COMMAND ----------

# DBTITLE 1,Most Common Themes
from collections import Counter

# Collect all themes
all_themes = []
for summary in all_summaries:
    all_themes.extend(summary.key_themes)

# Count theme frequency
theme_counts = Counter(all_themes)

print("Top 20 Most Common Themes:")
print("-" * 40)
for theme, count in theme_counts.most_common(20):
    print(f"  {theme}: {count}")

# COMMAND ----------

# DBTITLE 1,Sample Summaries by Level
print("=" * 60)
print("SAMPLE SUMMARIES BY LEVEL")
print("=" * 60)

for level in sorted(communities_by_level.keys()):
    level_summaries = [s for s in all_summaries if s.level == level]

    print(f"\n📊 LEVEL {level} SAMPLES:")
    print("-" * 40)

    # Show top 3 by member count
    sorted_summaries = sorted(level_summaries, key=lambda s: s.member_count, reverse=True)

    for summary in sorted_summaries[:3]:
        print(f"\n  {summary.community_id} ({summary.member_count} members):")
        print(f"    {summary.summary[:300]}...")
        if summary.key_themes:
            print(f"    Themes: {', '.join(summary.key_themes)}")
        if summary.key_entities:
            print(f"    Key entities: {', '.join(summary.key_entities)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Summaries to Delta Lake

# COMMAND ----------

# DBTITLE 1,Prepare Summary Data
# Convert summaries to dictionaries
summaries_data = []
for summary in all_summaries:
    summaries_data.append({
        "community_id": summary.community_id,
        "level": summary.level,
        "summary": summary.summary,
        "key_entities": json.dumps(summary.key_entities),
        "key_themes": json.dumps(summary.key_themes),
        "member_count": summary.member_count,
        "source_chunks": json.dumps(summary.source_chunks),
        "created_at": datetime.now().isoformat()
    })

print(f"Prepared {len(summaries_data)} summary records")

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
# Create DataFrame
summaries_df = spark.createDataFrame(summaries_data)

# Save to Delta Lake
summaries_df.write.format("delta").mode("overwrite").save(SUMMARIES_TABLE)
print(f"Saved {summaries_df.count()} summaries to {SUMMARIES_TABLE}")

# Display sample
display(summaries_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Index Summaries for Retrieval

# COMMAND ----------

# DBTITLE 1,Generate Summary Embeddings
from langchain_openai import AzureOpenAIEmbeddings

# Initialize embeddings model
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-01"
)

print("Generating embeddings for summaries...")

# Generate embeddings for all summaries
summary_texts = [s.summary for s in all_summaries]
summary_embeddings = embeddings.embed_documents(summary_texts)

print(f"Generated {len(summary_embeddings)} embeddings")
print(f"Embedding dimension: {len(summary_embeddings[0])}")

# COMMAND ----------

# DBTITLE 1,Prepare Indexed Data
# Combine summaries with embeddings
indexed_data = []
for i, summary in enumerate(all_summaries):
    indexed_data.append({
        "community_id": summary.community_id,
        "level": summary.level,
        "summary": summary.summary,
        "key_entities": json.dumps(summary.key_entities),
        "key_themes": json.dumps(summary.key_themes),
        "member_count": summary.member_count,
        "embedding": json.dumps(summary_embeddings[i]),  # Store as JSON
        "created_at": datetime.now().isoformat()
    })

# Save indexed data
indexed_df = spark.createDataFrame(indexed_data)
indexed_df.write.format("delta").mode("overwrite").save(SUMMARIES_INDEXED_TABLE)
print(f"Saved indexed summaries to {SUMMARIES_INDEXED_TABLE}")

# COMMAND ----------

# DBTITLE 1,Create Azure AI Search Index (Optional)
# Optional: Index summaries in Azure AI Search for hybrid retrieval
CREATE_SEARCH_INDEX = False  # Set to True to create index

if CREATE_SEARCH_INDEX:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SimpleField,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile
    )
    from azure.core.credentials import AzureKeyCredential

    # Create index schema
    index_name = "community-summaries"

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="community_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String),
        SearchableField(name="key_entities", type=SearchFieldDataType.String),
        SearchableField(name="key_themes", type=SearchFieldDataType.String),
        SimpleField(name="member_count", type=SearchFieldDataType.Int32, sortable=True),
        SearchField(
            name="summary_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,  # text-embedding-3-large
            vector_search_profile_name="vector-profile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ],
        profiles=[
            VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")
        ]
    )

    # Create index
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Created search index: {index_name}")

    # Upload documents
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    documents = []
    for i, summary in enumerate(all_summaries):
        documents.append({
            "id": summary.community_id,
            "community_id": summary.community_id,
            "level": summary.level,
            "summary": summary.summary,
            "key_entities": ", ".join(summary.key_entities),
            "key_themes": ", ".join(summary.key_themes),
            "member_count": summary.member_count,
            "summary_vector": summary_embeddings[i]
        })

    result = search_client.upload_documents(documents)
    print(f"Indexed {len(documents)} summaries in Azure AI Search")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary Retrieval Test

# COMMAND ----------

# DBTITLE 1,Test Summary Retrieval
# Test retrieval with a sample query
test_query = "project management and team collaboration"

# Generate query embedding
query_embedding = embeddings.embed_query(test_query)

# Simple cosine similarity search
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar summaries
similarities = []
for i, summary in enumerate(all_summaries):
    sim = cosine_similarity(query_embedding, summary_embeddings[i])
    similarities.append((summary, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{test_query}'")
print("\nTop 5 Most Relevant Communities:")
print("-" * 60)

for summary, sim in similarities[:5]:
    print(f"\n{summary.community_id} (similarity: {sim:.4f})")
    print(f"  Level: {summary.level}")
    print(f"  Members: {summary.member_count}")
    print(f"  Summary: {summary.summary[:200]}...")
    if summary.key_themes:
        print(f"  Themes: {', '.join(summary.key_themes[:3])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Phase 4 Summary
print("=" * 60)
print("PHASE 4.4: COMMUNITY SUMMARIZATION COMPLETE")
print("=" * 60)

print(f"\n📊 SUMMARIZATION RESULTS:")
print(f"  Total summaries: {len(all_summaries)}")
print(f"  Summaries by level:")
for level in sorted(communities_by_level.keys()):
    count = len([s for s in all_summaries if s.level == level])
    print(f"    Level {level}: {count}")
print(f"  Errors: {errors}")

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"  Summaries: {SUMMARIES_TABLE}")
print(f"  Indexed summaries: {SUMMARIES_INDEXED_TABLE}")

print(f"\n📝 SUMMARY QUALITY:")
print(f"  Average length: {avg_length:.0f} characters")
print(f"  Average themes per community: {avg_themes:.1f}")
print(f"  Total unique themes: {len(set(t for s in all_summaries for t in s.key_themes))}")

print("\n" + "=" * 60)
print("PHASE 4 GRAPHRAG CONSTRUCTION COMPLETE!")
print("=" * 60)

print("""
✅ COMPLETED COMPONENTS:
  1. Entity Extraction (GPT-4o + Pydantic)
  2. Knowledge Graph (Cosmos DB / In-Memory)
  3. Community Detection (Leiden Algorithm)
  4. Community Summarization (GPT-4o)

🎯 NEXT PHASE: P5 ReAct Agent (Weeks 9-10)
  - Integrate GraphRAG into retrieval pipeline
  - Implement ReAct reasoning loop
  - Multi-hop question answering
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Community Summaries:**
# MAGIC - Generated using GPT-4o for natural language quality
# MAGIC - Include key entities and themes for quick reference
# MAGIC - Organized by hierarchy level for multi-scale retrieval
# MAGIC
# MAGIC **Indexing Options:**
# MAGIC 1. **Delta Lake**: Full data storage with embeddings
# MAGIC 2. **Azure AI Search**: Production hybrid search with HNSW vectors
# MAGIC
# MAGIC **GraphRAG Retrieval Strategy:**
# MAGIC - Global queries → High-level (coarse) community summaries
# MAGIC - Specific queries → Low-level (fine) community summaries + entity lookup
# MAGIC - Hybrid: Vector similarity + keyword matching
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. P5: Build ReAct agent with LangGraph
# MAGIC 2. P5: Integrate GraphRAG + Vector retrieval
# MAGIC 3. P6: Evaluation with RAGAS framework
