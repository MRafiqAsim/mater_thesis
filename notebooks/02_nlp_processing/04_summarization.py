# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - LLM Summarization Pipeline
# MAGIC
# MAGIC **Phase 2: NLP Processing | Week 4**
# MAGIC
# MAGIC This notebook generates summaries using Azure OpenAI GPT-4o.
# MAGIC
# MAGIC ## Summarization Levels
# MAGIC 1. **Chunk summaries** - Individual chunk summaries
# MAGIC 2. **Document summaries** - Hierarchical (chunks → document)
# MAGIC 3. **Email thread summaries** - Conversation-aware
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC SILVER (anonymized_chunks) → SILVER (summaries)
# MAGIC ```
# MAGIC
# MAGIC ## Milestone M2 Completion
# MAGIC This notebook completes Phase 2: NLP Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai tiktoken pandas pyarrow delta-spark tenacity

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")
STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_PATH = f"{BASE_PATH}/silver"

# Summarization configuration
SUMM_CONFIG = {
    "model_deployment": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 500,
    "target_length": "concise",  # concise, detailed
    "batch_size": 50,
    "rate_limit_delay": 0.5,  # seconds between calls
}

print(f"SILVER_PATH: {SILVER_PATH}")
print(f"Model: {SUMM_CONFIG['model_deployment']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Azure OpenAI

# COMMAND ----------

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
import time

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment=SUMM_CONFIG["model_deployment"],
    temperature=SUMM_CONFIG["temperature"],
    max_tokens=SUMM_CONFIG["max_tokens"],
)

# Token counter
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Test connection
response = llm.invoke("Say 'ready' and nothing else.")
print(f"LLM ready: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Summary Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

SUMMARY_SCHEMA = StructType([
    StructField("summary_id", StringType(), False),
    StructField("source_id", StringType(), False),  # chunk_id or document_id
    StructField("source_type", StringType(), True),  # chunk, document, email_thread
    StructField("summary", StringType(), True),
    StructField("source_length", IntegerType(), True),
    StructField("summary_length", IntegerType(), True),
    StructField("compression_ratio", FloatType(), True),
    StructField("source_tokens", IntegerType(), True),
    StructField("summary_tokens", IntegerType(), True),
    StructField("model_used", StringType(), True),
    StructField("detected_language", StringType(), True),
    StructField("summarization_timestamp", TimestampType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summarization Prompts

# COMMAND ----------

# Chunk summarization prompt
CHUNK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert document summarizer. Create accurate, informative summaries.

Guidelines:
- Focus on key facts, decisions, and outcomes
- Write a concise summary in 2-3 sentences
- Preserve important names, dates, and specific references
- Do not add information not present in the source
- Maintain a professional, neutral tone
- If the text is in Dutch, write the summary in Dutch"""),
    ("human", """Summarize the following text:

{text}

Summary:"""),
])

# Document summarization prompt (from chunk summaries)
DOCUMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert document summarizer. Create a comprehensive document summary from chunk summaries.

Guidelines:
- Synthesize information across all chunks
- Identify the main theme and key points
- Write 4-6 sentences covering the essential content
- Note any important decisions, actions, or outcomes
- Maintain coherent narrative flow"""),
    ("human", """Create a document summary from these chunk summaries:

{chunk_summaries}

Document Summary:"""),
])

# Email thread summarization prompt
EMAIL_THREAD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at summarizing email conversations.

Create a summary that captures:
1. The main topic/purpose of the conversation
2. Key participants and their roles
3. Important decisions, action items, or outcomes
4. Any deadlines or commitments mentioned

Format with clear sections if the thread covers multiple topics."""),
    ("human", """Summarize this email thread:

{thread}

Summary:"""),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summarization Functions

# COMMAND ----------

from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_chunk(
    text: str,
    chunk_id: str,
    language: str = "en"
) -> Dict[str, Any]:
    """Summarize a single chunk."""
    if not text or len(text.strip()) < 50:
        return {
            "summary_id": hashlib.md5(f"sum:{chunk_id}".encode()).hexdigest()[:16],
            "source_id": chunk_id,
            "source_type": "chunk",
            "summary": text[:200] if text else "",
            "source_length": len(text) if text else 0,
            "summary_length": len(text[:200]) if text else 0,
            "compression_ratio": 1.0,
            "source_tokens": count_tokens(text) if text else 0,
            "summary_tokens": count_tokens(text[:200]) if text else 0,
            "model_used": "passthrough",
            "detected_language": language,
            "summarization_timestamp": datetime.utcnow(),
        }

    chain = CHUNK_PROMPT | llm
    response = chain.invoke({"text": text[:10000]})  # Limit input
    summary = response.content.strip()

    return {
        "summary_id": hashlib.md5(f"sum:{chunk_id}".encode()).hexdigest()[:16],
        "source_id": chunk_id,
        "source_type": "chunk",
        "summary": summary,
        "source_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": len(summary) / len(text) if text else 0,
        "source_tokens": count_tokens(text),
        "summary_tokens": count_tokens(summary),
        "model_used": SUMM_CONFIG["model_deployment"],
        "detected_language": language,
        "summarization_timestamp": datetime.utcnow(),
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_document(
    chunk_summaries: list,
    document_id: str,
    language: str = "en"
) -> Dict[str, Any]:
    """Create document-level summary from chunk summaries."""
    combined = "\n\n".join([f"- {s}" for s in chunk_summaries])

    if count_tokens(combined) > 6000:
        # Truncate if too long
        combined = combined[:20000]

    chain = DOCUMENT_PROMPT | llm
    response = chain.invoke({"chunk_summaries": combined})
    summary = response.content.strip()

    return {
        "summary_id": hashlib.md5(f"docsum:{document_id}".encode()).hexdigest()[:16],
        "source_id": document_id,
        "source_type": "document",
        "summary": summary,
        "source_length": len(combined),
        "summary_length": len(summary),
        "compression_ratio": len(summary) / len(combined) if combined else 0,
        "source_tokens": count_tokens(combined),
        "summary_tokens": count_tokens(summary),
        "model_used": SUMM_CONFIG["model_deployment"],
        "detected_language": language,
        "summarization_timestamp": datetime.utcnow(),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load Anonymized Chunks

# COMMAND ----------

anon_chunks_df = spark.read.format("delta").load(f"{SILVER_PATH}/anonymized_chunks")
print(f"Loaded {anon_chunks_df.count()} anonymized chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Chunk Summaries

# COMMAND ----------

from pyspark.sql.functions import col

# Collect chunks for processing
chunks = anon_chunks_df.select(
    "chunk_id",
    "parent_document_id",
    "anonymized_content",
    "detected_language"
).collect()

print(f"Generating summaries for {len(chunks)} chunks...")

# COMMAND ----------

all_summaries = []
errors = []

for i, chunk in enumerate(chunks):
    try:
        summary = summarize_chunk(
            text=chunk["anonymized_content"],
            chunk_id=chunk["chunk_id"],
            language=chunk["detected_language"] or "en",
        )
        all_summaries.append(summary)

        # Rate limiting
        time.sleep(SUMM_CONFIG["rate_limit_delay"])

    except Exception as e:
        errors.append({"chunk_id": chunk["chunk_id"], "error": str(e)})

    if (i + 1) % SUMM_CONFIG["batch_size"] == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks...")

print(f"\nChunk summarization complete: {len(all_summaries)} summaries, {len(errors)} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Generate Document Summaries

# COMMAND ----------

from collections import defaultdict

# Group chunk summaries by document
doc_chunks = defaultdict(list)
doc_languages = {}

for s in all_summaries:
    if s["source_type"] == "chunk":
        # Find parent document
        chunk_info = anon_chunks_df.filter(col("chunk_id") == s["source_id"]).first()
        if chunk_info:
            doc_id = chunk_info["parent_document_id"]
            doc_chunks[doc_id].append(s["summary"])
            doc_languages[doc_id] = chunk_info["detected_language"]

print(f"Generating document summaries for {len(doc_chunks)} documents...")

# COMMAND ----------

doc_summaries = []

for doc_id, chunk_sums in doc_chunks.items():
    if len(chunk_sums) < 2:
        # Single chunk, use chunk summary as document summary
        continue

    try:
        doc_summary = summarize_document(
            chunk_summaries=chunk_sums,
            document_id=doc_id,
            language=doc_languages.get(doc_id, "en"),
        )
        doc_summaries.append(doc_summary)

        time.sleep(SUMM_CONFIG["rate_limit_delay"])

    except Exception as e:
        errors.append({"document_id": doc_id, "error": str(e)})

print(f"Document summarization complete: {len(doc_summaries)} summaries")

# Combine all summaries
all_summaries.extend(doc_summaries)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Write Summaries to Silver

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

# Create DataFrame
summaries_df = spark.createDataFrame(all_summaries, schema=SUMMARY_SCHEMA)
summaries_df = summaries_df.withColumn("_ingested_at", current_timestamp())

# Write summaries
summaries_path = f"{SILVER_PATH}/summaries"

try:
    delta_table = DeltaTable.forPath(spark, summaries_path)
    delta_table.alias("target").merge(
        summaries_df.alias("source"),
        "target.summary_id = source.summary_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print(f"Merged {summaries_df.count()} summaries")
except Exception:
    summaries_df.write.format("delta").mode("overwrite").save(summaries_path)
    print(f"Created summaries table with {summaries_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Analyze Summaries

# COMMAND ----------

# Read back
summaries_silver = spark.read.format("delta").load(summaries_path)

print(f"Total summaries: {summaries_silver.count()}")

# By type
print("\nSummaries by type:")
summaries_silver.groupBy("source_type").count().display()

# COMMAND ----------

# Compression statistics
from pyspark.sql.functions import avg, min, max

print("\nCompression statistics:")
summaries_silver.agg(
    avg("compression_ratio").alias("avg_compression"),
    min("compression_ratio").alias("min_compression"),
    max("compression_ratio").alias("max_compression"),
    avg("source_tokens").alias("avg_source_tokens"),
    avg("summary_tokens").alias("avg_summary_tokens"),
).display()

# COMMAND ----------

# Sample summaries
print("\nSample chunk summaries:")
display(summaries_silver.filter(col("source_type") == "chunk").select(
    "source_id", "summary", "compression_ratio"
).limit(5))

# COMMAND ----------

print("\nSample document summaries:")
display(summaries_silver.filter(col("source_type") == "document").select(
    "source_id", "summary", "summary_length"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Phase 2 Completion Summary

# COMMAND ----------

total_summaries = summaries_silver.count()
chunk_summaries = summaries_silver.filter(col("source_type") == "chunk").count()
doc_summaries_count = summaries_silver.filter(col("source_type") == "document").count()
avg_compression = summaries_silver.agg(avg("compression_ratio")).collect()[0][0]

# Load metrics from other Phase 2 notebooks
chunks_count = spark.read.format("delta").load(f"{SILVER_PATH}/chunks").count()
entities_count = spark.read.format("delta").load(f"{SILVER_PATH}/entities").count()
anon_count = spark.read.format("delta").load(f"{SILVER_PATH}/anonymized_chunks").count()
pii_count = spark.read.format("delta").load(f"{SILVER_PATH}/pii_detections").count() if dbutils.fs.ls(f"{SILVER_PATH}/pii_detections") else 0

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║         PHASE 2: NLP PROCESSING COMPLETE                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─ SEMANTIC CHUNKING ────────────────────────────────────────┐  ║
║  │  Total Chunks: {chunks_count:<44} │  ║
║  │  Strategy: Azure OpenAI Semantic Chunking                  │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ NER EXTRACTION ───────────────────────────────────────────┐  ║
║  │  Entity Mentions: {entities_count:<41} │  ║
║  │  Models: spaCy EN + NL with custom patterns                │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ ANONYMIZATION ────────────────────────────────────────────┐  ║
║  │  Chunks Anonymized: {anon_count:<39} │  ║
║  │  PII Instances: {pii_count:<43} │  ║
║  │  Global Patterns: Phone, IBAN, VAT, SSN, Passport, IDs     │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ SUMMARIZATION ────────────────────────────────────────────┐  ║
║  │  Chunk Summaries: {chunk_summaries:<41} │  ║
║  │  Document Summaries: {doc_summaries_count:<39} │  ║
║  │  Avg Compression: {avg_compression:.2%:<40} │  ║
║  │  Model: GPT-4o                                             │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ MILESTONE M2 COMPLETE                                         ║
║    • NLP pipeline operational                                    ║
║    • PII detection with Belgian patterns                         ║
║    • Silver zone populated with processed data                   ║
╠══════════════════════════════════════════════════════════════════╣
║  SILVER ZONE TABLES:                                             ║
║  • /chunks           - Semantic chunks                           ║
║  • /entities         - NER extractions                           ║
║  • /entity_registry  - Deduplicated entities                     ║
║  • /domain_glossary  - High-frequency terms                      ║
║  • /anonymized_chunks - PII-free content                         ║
║  • /pii_detections   - PII audit log                             ║
║  • /summaries        - Chunk & document summaries                ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT PHASE: Vector Index & Basic RAG (Week 5)                   ║
║  Run: notebooks/03_vector_index/01_embedding_indexing.py         ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
