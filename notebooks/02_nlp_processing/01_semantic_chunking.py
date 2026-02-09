# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Semantic Chunking Pipeline
# MAGIC
# MAGIC **Phase 2: NLP Processing | Week 3**
# MAGIC
# MAGIC This notebook implements intelligent document chunking using Azure OpenAI embeddings.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Split documents at semantic boundaries (not just character counts)
# MAGIC - Preserve context across chunks with overlap
# MAGIC - Optimize chunk sizes for embedding and retrieval
# MAGIC - Write chunks to Silver zone
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC BRONZE (documents, emails) → SILVER (semantic chunks)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai langchain-experimental tiktoken pandas pyarrow delta-spark

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Get Azure credentials
OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")
STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
BRONZE_PATH = f"{BASE_PATH}/bronze"
SILVER_PATH = f"{BASE_PATH}/silver"

# Chunking configuration
CHUNK_CONFIG = {
    "use_semantic": True,
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 95.0,
    "chunk_size": 1000,  # Fallback
    "chunk_overlap": 200,
    "max_tokens_per_chunk": 512,
    "min_tokens_per_chunk": 50,
}

print(f"BRONZE: {BRONZE_PATH}")
print(f"SILVER: {SILVER_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Semantic Chunker

# COMMAND ----------

from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
)

# Initialize semantic chunker
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type=CHUNK_CONFIG["breakpoint_threshold_type"],
    breakpoint_threshold_amount=CHUNK_CONFIG["breakpoint_threshold_amount"],
)

# Fallback chunker
fallback_chunker = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CONFIG["chunk_size"],
    chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Token counter
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

print("Chunkers initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Chunk Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType

CHUNK_SCHEMA = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("parent_document_id", StringType(), False),
    StructField("parent_type", StringType(), True),  # email, document
    StructField("chunk_index", IntegerType(), True),
    StructField("total_chunks", IntegerType(), True),
    StructField("content", StringType(), True),
    StructField("content_length", IntegerType(), True),
    StructField("token_count", IntegerType(), True),
    StructField("start_char", IntegerType(), True),
    StructField("end_char", IntegerType(), True),
    StructField("chunking_strategy", StringType(), True),
    StructField("detected_language", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Chunking Functions

# COMMAND ----------

from typing import List, Dict, Any
from datetime import datetime
import hashlib

def chunk_text(
    text: str,
    document_id: str,
    parent_type: str,
    language: str,
    source_file: str,
    use_semantic: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk text using semantic or fallback chunker.

    Returns list of chunk dictionaries.
    """
    if not text or len(text.strip()) < CHUNK_CONFIG["min_tokens_per_chunk"]:
        # Return as single chunk if too short
        return [{
            "chunk_id": hashlib.md5(f"{document_id}:0".encode()).hexdigest()[:16],
            "parent_document_id": document_id,
            "parent_type": parent_type,
            "chunk_index": 0,
            "total_chunks": 1,
            "content": text,
            "content_length": len(text),
            "token_count": count_tokens(text),
            "start_char": 0,
            "end_char": len(text),
            "chunking_strategy": "single",
            "detected_language": language,
            "source_file": source_file,
            "ingestion_timestamp": datetime.utcnow(),
        }]

    try:
        # Try semantic chunking
        if use_semantic:
            from langchain_core.documents import Document
            doc = Document(page_content=text)
            chunks = semantic_chunker.split_documents([doc])
            strategy = "semantic"
        else:
            raise Exception("Fallback requested")

    except Exception as e:
        # Fallback to recursive
        print(f"Semantic chunking failed for {document_id}, using fallback: {str(e)[:50]}")
        from langchain_core.documents import Document
        doc = Document(page_content=text)
        chunks = fallback_chunker.split_documents([doc])
        strategy = "recursive"

    # Convert to dictionaries
    result = []
    char_offset = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        token_count = count_tokens(chunk_text)

        # Skip tiny chunks
        if token_count < CHUNK_CONFIG["min_tokens_per_chunk"]:
            continue

        # Find position in original
        start_char = text.find(chunk_text[:50], char_offset)
        if start_char == -1:
            start_char = char_offset
        end_char = start_char + len(chunk_text)
        char_offset = end_char

        chunk_id = hashlib.md5(f"{document_id}:{i}:{chunk_text[:50]}".encode()).hexdigest()[:16]

        result.append({
            "chunk_id": chunk_id,
            "parent_document_id": document_id,
            "parent_type": parent_type,
            "chunk_index": i,
            "total_chunks": len(chunks),  # Will update later
            "content": chunk_text,
            "content_length": len(chunk_text),
            "token_count": token_count,
            "start_char": start_char,
            "end_char": end_char,
            "chunking_strategy": strategy,
            "detected_language": language,
            "source_file": source_file,
            "ingestion_timestamp": datetime.utcnow(),
        })

    # Update total_chunks
    for chunk in result:
        chunk["total_chunks"] = len(result)

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Bronze Data

# COMMAND ----------

# Load documents
docs_df = spark.read.format("delta").load(f"{BRONZE_PATH}/documents")
print(f"Loaded {docs_df.count()} documents")

# Load emails
emails_df = spark.read.format("delta").load(f"{BRONZE_PATH}/emails")
print(f"Loaded {emails_df.count()} emails")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Process Documents

# COMMAND ----------

from pyspark.sql.functions import col

# Collect documents for processing
# Note: For large datasets, use Spark UDF instead of collect
documents = docs_df.select(
    "document_id",
    "content",
    "detected_language",
    "filename"
).collect()

print(f"Processing {len(documents)} documents...")

# COMMAND ----------

all_chunks = []
errors = []

for i, doc in enumerate(documents):
    try:
        doc_chunks = chunk_text(
            text=doc["content"],
            document_id=doc["document_id"],
            parent_type="document",
            language=doc["detected_language"] or "en",
            source_file=doc["filename"],
            use_semantic=CHUNK_CONFIG["use_semantic"],
        )
        all_chunks.extend(doc_chunks)

    except Exception as e:
        errors.append({"document_id": doc["document_id"], "error": str(e)})

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(documents)} documents, {len(all_chunks)} chunks created")

print(f"\nDocument chunking complete: {len(all_chunks)} chunks, {len(errors)} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Process Emails

# COMMAND ----------

# Collect emails
emails = emails_df.select(
    "message_id",
    "body_text",
    "detected_language",
    "source_file"
).collect()

print(f"Processing {len(emails)} emails...")

# COMMAND ----------

for i, email in enumerate(emails):
    try:
        email_chunks = chunk_text(
            text=email["body_text"],
            document_id=email["message_id"],
            parent_type="email",
            language=email["detected_language"] or "en",
            source_file=email["source_file"] or "email",
            use_semantic=CHUNK_CONFIG["use_semantic"],
        )
        all_chunks.extend(email_chunks)

    except Exception as e:
        errors.append({"document_id": email["message_id"], "error": str(e)})

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(emails)} emails, {len(all_chunks)} total chunks")

print(f"\nEmail chunking complete: {len(all_chunks)} total chunks, {len(errors)} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write Chunks to Silver Zone

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

# Create DataFrame
chunks_df = spark.createDataFrame(all_chunks, schema=CHUNK_SCHEMA)
chunks_df = chunks_df.withColumn("_ingested_at", current_timestamp())

# Write to Silver
chunks_path = f"{SILVER_PATH}/chunks"

try:
    delta_table = DeltaTable.forPath(spark, chunks_path)
    delta_table.alias("target").merge(
        chunks_df.alias("source"),
        "target.chunk_id = source.chunk_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print(f"Merged {chunks_df.count()} chunks")
except Exception:
    chunks_df.write.format("delta").mode("overwrite").save(chunks_path)
    print(f"Created chunks table with {chunks_df.count()} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verify & Analyze Chunks

# COMMAND ----------

# Read back
chunks_silver = spark.read.format("delta").load(chunks_path)

print(f"Total chunks in Silver: {chunks_silver.count()}")

# By parent type
print("\nChunks by parent type:")
chunks_silver.groupBy("parent_type").count().display()

# By chunking strategy
print("\nChunks by strategy:")
chunks_silver.groupBy("chunking_strategy").count().display()

# By language
print("\nChunks by language:")
chunks_silver.groupBy("detected_language").count().display()

# COMMAND ----------

# Token distribution
from pyspark.sql.functions import avg, min, max, sum, stddev

print("Token statistics:")
chunks_silver.agg(
    avg("token_count").alias("avg_tokens"),
    min("token_count").alias("min_tokens"),
    max("token_count").alias("max_tokens"),
    stddev("token_count").alias("stddev_tokens"),
    sum("token_count").alias("total_tokens"),
).display()

# COMMAND ----------

# Chunks per document distribution
print("Chunks per document:")
chunks_silver.groupBy("parent_document_id").count().groupBy("count").count().orderBy("count").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

total_chunks = chunks_silver.count()
total_tokens = chunks_silver.agg(sum("token_count")).collect()[0][0]
avg_tokens = chunks_silver.agg(avg("token_count")).collect()[0][0]

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║            SEMANTIC CHUNKING COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Chunks Created    : {total_chunks:<35} ║
║  Total Tokens            : {total_tokens:<35} ║
║  Average Tokens/Chunk    : {avg_tokens:.1f:<34} ║
║  Silver Table Location   : {SILVER_PATH}/chunks                 ║
╠══════════════════════════════════════════════════════════════════╣
║  CHUNKING STRATEGY:                                              ║
║  • Semantic chunking with Azure OpenAI embeddings                ║
║  • Breakpoint threshold: {CHUNK_CONFIG['breakpoint_threshold_amount']}th percentile                    ║
║  • Target chunk size: {CHUNK_CONFIG['max_tokens_per_chunk']} tokens                            ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS:                                                     ║
║  1. Run 02_ner_extraction.py for entity extraction               ║
║  2. Run 03_anonymization.py for PII removal                      ║
║  3. Run 04_summarization.py for chunk summaries                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
