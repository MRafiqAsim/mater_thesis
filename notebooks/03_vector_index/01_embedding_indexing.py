# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Embedding Generation & Azure AI Search Indexing
# MAGIC
# MAGIC **Phase 3: Vector Index & Basic RAG | Week 5**
# MAGIC
# MAGIC This notebook generates embeddings and indexes chunks to Azure AI Search.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Generate embeddings using text-embedding-3-large (3072 dimensions)
# MAGIC - Create Azure AI Search index with HNSW vector configuration
# MAGIC - Index all chunks with embeddings and metadata
# MAGIC - Enable hybrid search (vector + keyword)
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC SILVER (anonymized_chunks, summaries) → Azure AI Search Index
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai azure-search-documents azure-identity tiktoken pandas pyarrow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Azure credentials
OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")
SEARCH_ENDPOINT = dbutils.secrets.get("azure-search", "endpoint")
SEARCH_KEY = dbutils.secrets.get("azure-search", "api-key")
STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_PATH = f"{BASE_PATH}/silver"
GOLD_PATH = f"{BASE_PATH}/gold"

# Index configuration
INDEX_CONFIG = {
    "index_name": "knowledge-chunks",
    "vector_dimensions": 3072,  # text-embedding-3-large
    "hnsw_m": 4,
    "hnsw_ef_construction": 400,
    "hnsw_ef_search": 500,
    "batch_size": 100,
}

print(f"Search Endpoint: {SEARCH_ENDPOINT}")
print(f"Index Name: {INDEX_CONFIG['index_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Azure AI Search Client

# COMMAND ----------

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)

# Initialize clients
credential = AzureKeyCredential(SEARCH_KEY)
index_client = SearchIndexClient(SEARCH_ENDPOINT, credential)

print("Azure AI Search client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Search Index with Vector Configuration

# COMMAND ----------

def create_search_index():
    """Create or update the search index with vector and semantic configuration."""

    # Define fields
    fields = [
        # Key field
        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # Content fields
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name="summary",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Vector field (3072 dimensions for text-embedding-3-large)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=INDEX_CONFIG["vector_dimensions"],
            vector_search_profile_name="hnsw-profile",
        ),
        # Metadata fields
        SearchField(
            name="parent_document_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="parent_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="detected_language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="token_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SearchField(
            name="entities",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="indexed_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
    ]

    # Vector search configuration (HNSW)
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters={
                    "m": INDEX_CONFIG["hnsw_m"],
                    "efConstruction": INDEX_CONFIG["hnsw_ef_construction"],
                    "efSearch": INDEX_CONFIG["hnsw_ef_search"],
                    "metric": "cosine",
                },
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            ),
        ],
    )

    # Semantic configuration for reranking
    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            title_fields=[SemanticField(field_name="summary")],
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create index
    index = SearchIndex(
        name=INDEX_CONFIG["index_name"],
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    index_client.create_or_update_index(index)
    print(f"Index '{INDEX_CONFIG['index_name']}' created/updated successfully")

    return True

# Create the index
create_search_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Initialize Embedding Model

# COMMAND ----------

from langchain_openai import AzureOpenAIEmbeddings
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
)

# Token counter
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_embedding(text: str) -> list:
    """Generate embedding with retry logic."""
    # Truncate if too long (8191 token limit for embedding model)
    if count_tokens(text) > 8000:
        # Truncate by characters (rough approximation)
        text = text[:30000]
    return embeddings.embed_query(text)

# Test embedding
test_embedding = generate_embedding("Test embedding generation")
print(f"Embedding dimensions: {len(test_embedding)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Chunks and Summaries from Silver

# COMMAND ----------

# Load anonymized chunks
chunks_df = spark.read.format("delta").load(f"{SILVER_PATH}/anonymized_chunks")
print(f"Loaded {chunks_df.count()} anonymized chunks")

# Load summaries
summaries_df = spark.read.format("delta").load(f"{SILVER_PATH}/summaries")
chunk_summaries = summaries_df.filter("source_type = 'chunk'")
print(f"Loaded {chunk_summaries.count()} chunk summaries")

# Load entities (for metadata enrichment)
try:
    entities_df = spark.read.format("delta").load(f"{SILVER_PATH}/entities")
    print(f"Loaded {entities_df.count()} entities")
except Exception:
    entities_df = None
    print("No entities table found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Prepare Documents for Indexing

# COMMAND ----------

from pyspark.sql.functions import col, collect_list

# Join chunks with summaries
chunks_with_summary = chunks_df.alias("c").join(
    chunk_summaries.alias("s"),
    col("c.chunk_id") == col("s.source_id"),
    "left"
).select(
    col("c.chunk_id"),
    col("c.parent_document_id"),
    col("c.anonymized_content").alias("content"),
    col("s.summary"),
    col("c.detected_language"),
    col("c.pii_count"),
)

# Get original chunk metadata
original_chunks = spark.read.format("delta").load(f"{SILVER_PATH}/chunks")
chunks_full = chunks_with_summary.alias("cws").join(
    original_chunks.alias("oc"),
    col("cws.chunk_id") == col("oc.chunk_id"),
    "left"
).select(
    col("cws.chunk_id"),
    col("cws.parent_document_id"),
    col("cws.content"),
    col("cws.summary"),
    col("cws.detected_language"),
    col("oc.parent_type"),
    col("oc.source_file"),
    col("oc.chunk_index"),
    col("oc.token_count"),
)

# Add entities if available
if entities_df is not None:
    # Aggregate entities per chunk
    chunk_entities = entities_df.groupBy("chunk_id").agg(
        collect_list("normalized_text").alias("entities")
    )

    chunks_full = chunks_full.join(
        chunk_entities,
        "chunk_id",
        "left"
    )
else:
    from pyspark.sql.functions import array, lit
    chunks_full = chunks_full.withColumn("entities", array())

print(f"Prepared {chunks_full.count()} documents for indexing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Embeddings and Index

# COMMAND ----------

from datetime import datetime
import time

# Collect documents
documents = chunks_full.collect()
print(f"Processing {len(documents)} documents...")

# Initialize search client
search_client = SearchClient(
    SEARCH_ENDPOINT,
    INDEX_CONFIG["index_name"],
    credential
)

# Process in batches
batch_size = INDEX_CONFIG["batch_size"]
success_count = 0
error_count = 0
all_docs = []

for i, doc in enumerate(documents):
    try:
        # Prepare document
        content = doc["content"] or ""

        # Generate embedding
        if content:
            embedding = generate_embedding(content)
        else:
            embedding = [0.0] * INDEX_CONFIG["vector_dimensions"]

        # Build document for indexing
        index_doc = {
            "chunk_id": doc["chunk_id"],
            "content": content,
            "summary": doc["summary"] or "",
            "content_vector": embedding,
            "parent_document_id": doc["parent_document_id"],
            "parent_type": doc["parent_type"] or "document",
            "detected_language": doc["detected_language"] or "en",
            "source_file": doc["source_file"] or "",
            "chunk_index": doc["chunk_index"] or 0,
            "token_count": doc["token_count"] or 0,
            "entities": doc["entities"] or [],
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }

        all_docs.append(index_doc)

        # Upload in batches
        if len(all_docs) >= batch_size:
            result = search_client.upload_documents(all_docs)
            batch_success = sum(1 for r in result if r.succeeded)
            success_count += batch_success
            error_count += len(all_docs) - batch_success
            all_docs = []
            print(f"Indexed batch, total success: {success_count}, errors: {error_count}")

            # Rate limiting
            time.sleep(0.5)

    except Exception as e:
        print(f"Error processing doc {doc['chunk_id']}: {str(e)[:100]}")
        error_count += 1

# Upload remaining documents
if all_docs:
    result = search_client.upload_documents(all_docs)
    batch_success = sum(1 for r in result if r.succeeded)
    success_count += batch_success
    error_count += len(all_docs) - batch_success

print(f"\nIndexing complete: {success_count} success, {error_count} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify Index

# COMMAND ----------

# Get index statistics
index_info = index_client.get_index(INDEX_CONFIG["index_name"])
print(f"Index: {index_info.name}")
print(f"Fields: {len(index_info.fields)}")

# Test search
from azure.search.documents.models import VectorizedQuery

# Test vector search
test_query = "project timeline and milestones"
test_embedding = generate_embedding(test_query)

vector_query = VectorizedQuery(
    vector=test_embedding,
    k_nearest_neighbors=5,
    fields="content_vector"
)

results = search_client.search(
    search_text=test_query,
    vector_queries=[vector_query],
    top=5,
    select=["chunk_id", "content", "parent_type", "source_file"],
)

print(f"\nTest search results for: '{test_query}'")
for i, result in enumerate(results, 1):
    print(f"\n[{i}] Score: {result['@search.score']:.4f}")
    print(f"    Source: {result.get('source_file', 'N/A')}")
    print(f"    Content: {result.get('content', '')[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Test Different Search Types

# COMMAND ----------

def test_search(query: str, search_type: str = "hybrid"):
    """Test different search types."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Search Type: {search_type}")
    print(f"{'='*60}")

    if search_type == "vector":
        # Pure vector search
        query_embedding = generate_embedding(query)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=5,
            fields="content_vector"
        )
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=5,
            select=["chunk_id", "content", "source_file"],
        )

    elif search_type == "keyword":
        # Pure keyword search
        results = search_client.search(
            search_text=query,
            top=5,
            select=["chunk_id", "content", "source_file"],
        )

    else:  # hybrid
        # Hybrid search
        query_embedding = generate_embedding(query)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=5,
            fields="content_vector"
        )
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=5,
            select=["chunk_id", "content", "source_file"],
            query_type="semantic",
            semantic_configuration_name="semantic-config",
        )

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['@search.score']:.4f}")
        content = result.get('content', '')[:200]
        print(f"    {content}...")

# Test different search types
test_query = "What are the key decisions made in the project?"

test_search(test_query, "vector")
test_search(test_query, "keyword")
test_search(test_query, "hybrid")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Indexing Metadata

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# Save indexing metadata
indexing_metadata = {
    "index_name": INDEX_CONFIG["index_name"],
    "documents_indexed": success_count,
    "errors": error_count,
    "vector_dimensions": INDEX_CONFIG["vector_dimensions"],
    "embedding_model": "text-embedding-3-large",
    "indexed_at": datetime.utcnow().isoformat(),
}

# Save to Gold zone
import json
metadata_path = f"{GOLD_PATH}/index_metadata/metadata.json"
dbutils.fs.put(metadata_path, json.dumps(indexing_metadata, indent=2), overwrite=True)
print(f"Saved indexing metadata to {metadata_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary

# COMMAND ----------

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║          VECTOR INDEXING COMPLETE                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Index Name          : {INDEX_CONFIG['index_name']:<41} ║
║  Documents Indexed   : {success_count:<41} ║
║  Indexing Errors     : {error_count:<41} ║
╠══════════════════════════════════════════════════════════════════╣
║  VECTOR CONFIGURATION:                                           ║
║  • Embedding Model: text-embedding-3-large                       ║
║  • Dimensions: {INDEX_CONFIG['vector_dimensions']}                                           ║
║  • Algorithm: HNSW (m={INDEX_CONFIG['hnsw_m']}, efConstruction={INDEX_CONFIG['hnsw_ef_construction']})             ║
║  • Metric: Cosine Similarity                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  SEARCH CAPABILITIES:                                            ║
║  • Vector search (semantic similarity)                           ║
║  • Keyword search (BM25)                                         ║
║  • Hybrid search (vector + keyword)                              ║
║  • Semantic ranking (reranker)                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS:                                                     ║
║  1. Run 02_basic_rag.py to implement RAG chain                   ║
║  2. Run 03_baseline_metrics.py to measure baseline performance   ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
