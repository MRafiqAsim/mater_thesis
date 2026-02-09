# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Named Entity Recognition (NER) Pipeline
# MAGIC
# MAGIC **Phase 2: NLP Processing | Week 3**
# MAGIC
# MAGIC This notebook extracts named entities from chunks using spaCy multilingual models.
# MAGIC
# MAGIC ## Entity Types
# MAGIC - PERSON - People names
# MAGIC - ORG - Organizations, companies
# MAGIC - GPE - Geopolitical entities (countries, cities)
# MAGIC - DATE - Dates and time expressions
# MAGIC - MONEY - Monetary values
# MAGIC - PROJECT - Project identifiers (custom)
# MAGIC - TECHNOLOGY - Tech terms (custom)
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC SILVER (chunks) → SILVER (chunks + entities) + SILVER (entity registry)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install spacy pandas pyarrow delta-spark

# COMMAND ----------

# Download spaCy models
# MAGIC %sh
# MAGIC python -m spacy download en_core_web_sm
# MAGIC python -m spacy download nl_core_news_sm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_PATH = f"{BASE_PATH}/silver"

# NER Configuration
NER_CONFIG = {
    "english_model": "en_core_web_sm",  # Use _trf for transformer model if available
    "dutch_model": "nl_core_news_sm",
    "entity_types": [
        "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
        "MONEY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "FAC", "NORP"
    ],
    "min_entity_length": 2,
    "batch_size": 100,
}

print(f"SILVER_PATH: {SILVER_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize spaCy Models

# COMMAND ----------

import spacy
from spacy.pipeline import EntityRuler

# Load models
nlp_en = spacy.load(NER_CONFIG["english_model"])
nlp_nl = spacy.load(NER_CONFIG["dutch_model"])

# Add custom patterns
custom_patterns = [
    # Project patterns
    {"label": "PROJECT", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}-\d{3,6}$"}}]},
    {"label": "PROJECT", "pattern": [{"LOWER": "project"}, {"IS_TITLE": True}]},

    # Technology patterns
    {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": [
        "azure", "aws", "python", "java", "kubernetes", "docker",
        "tensorflow", "pytorch", "spark", "databricks", "openai",
        "langchain", "gpt", "api", "sql", "nosql", "graphdb"
    ]}}]},

    # Document reference
    {"label": "DOCUMENT_REF", "pattern": [{"TEXT": {"REGEX": r"^DOC-\d{4,8}$"}}]},
]

# Add entity ruler to English model
if "entity_ruler" not in nlp_en.pipe_names:
    ruler_en = nlp_en.add_pipe("entity_ruler", before="ner")
    ruler_en.add_patterns(custom_patterns)

# Add entity ruler to Dutch model
if "entity_ruler" not in nlp_nl.pipe_names:
    ruler_nl = nlp_nl.add_pipe("entity_ruler", before="ner")
    ruler_nl.add_patterns(custom_patterns)

print(f"English model: {nlp_en.meta['name']}")
print(f"Dutch model: {nlp_nl.meta['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Entity Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, TimestampType

# Schema for extracted entities
ENTITY_SCHEMA = StructType([
    StructField("entity_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("parent_document_id", StringType(), False),
    StructField("text", StringType(), True),
    StructField("normalized_text", StringType(), True),
    StructField("label", StringType(), True),
    StructField("start_char", IntegerType(), True),
    StructField("end_char", IntegerType(), True),
    StructField("confidence", FloatType(), True),
    StructField("language", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
])

# Schema for entity registry (deduplicated)
ENTITY_REGISTRY_SCHEMA = StructType([
    StructField("canonical_id", StringType(), False),
    StructField("canonical_name", StringType(), True),
    StructField("label", StringType(), True),
    StructField("mention_count", IntegerType(), True),
    StructField("document_count", IntegerType(), True),
    StructField("variants", ArrayType(StringType()), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. NER Extraction Functions

# COMMAND ----------

from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib
import re

def normalize_entity(text: str, label: str) -> str:
    """Normalize entity text for consistency."""
    normalized = " ".join(text.split())

    if label == "PERSON":
        normalized = normalized.title()
    elif label in ("DATE", "TIME"):
        pass  # Keep as-is
    elif label == "MONEY":
        normalized = re.sub(r'\s+', '', normalized)

    return normalized

def extract_entities(
    text: str,
    chunk_id: str,
    parent_document_id: str,
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.

    Returns list of entity dictionaries.
    """
    # Select model based on language
    nlp = nlp_en if language == "en" else nlp_nl

    # Process text
    doc = nlp(text[:100000])  # Limit for performance

    entities = []
    for ent in doc.ents:
        # Filter by entity type
        if ent.label_ not in NER_CONFIG["entity_types"] and ent.label_ not in ["PROJECT", "TECHNOLOGY", "DOCUMENT_REF"]:
            continue

        # Filter by length
        if len(ent.text.strip()) < NER_CONFIG["min_entity_length"]:
            continue

        # Generate entity ID
        entity_id = hashlib.md5(
            f"{chunk_id}:{ent.start_char}:{ent.end_char}:{ent.text}".encode()
        ).hexdigest()[:16]

        entities.append({
            "entity_id": entity_id,
            "chunk_id": chunk_id,
            "parent_document_id": parent_document_id,
            "text": ent.text,
            "normalized_text": normalize_entity(ent.text, ent.label_),
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "confidence": 1.0,  # spaCy doesn't provide confidence
            "language": language,
            "ingestion_timestamp": datetime.utcnow(),
        })

    return entities

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Chunks from Silver

# COMMAND ----------

chunks_df = spark.read.format("delta").load(f"{SILVER_PATH}/chunks")
print(f"Loaded {chunks_df.count()} chunks")

# Sample by language
print("\nChunks by language:")
chunks_df.groupBy("detected_language").count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Extract Entities from Chunks

# COMMAND ----------

from pyspark.sql.functions import col

# Collect chunks for processing
chunks = chunks_df.select(
    "chunk_id",
    "parent_document_id",
    "content",
    "detected_language"
).collect()

print(f"Processing {len(chunks)} chunks...")

# COMMAND ----------

all_entities = []
entity_registry = defaultdict(lambda: {
    "canonical_name": None,
    "label": None,
    "mention_count": 0,
    "documents": set(),
    "variants": set(),
})

errors = []

for i, chunk in enumerate(chunks):
    try:
        chunk_entities = extract_entities(
            text=chunk["content"],
            chunk_id=chunk["chunk_id"],
            parent_document_id=chunk["parent_document_id"],
            language=chunk["detected_language"] or "en",
        )

        # Add to all entities
        all_entities.extend(chunk_entities)

        # Update registry
        for ent in chunk_entities:
            canonical_key = hashlib.md5(
                f"{ent['label']}:{ent['normalized_text'].lower()}".encode()
            ).hexdigest()[:12]

            entity_registry[canonical_key]["canonical_name"] = ent["normalized_text"]
            entity_registry[canonical_key]["label"] = ent["label"]
            entity_registry[canonical_key]["mention_count"] += 1
            entity_registry[canonical_key]["documents"].add(ent["parent_document_id"])
            entity_registry[canonical_key]["variants"].add(ent["text"])

    except Exception as e:
        errors.append({"chunk_id": chunk["chunk_id"], "error": str(e)})

    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks, {len(all_entities)} entities extracted")

print(f"\nNER complete: {len(all_entities)} entities, {len(entity_registry)} unique, {len(errors)} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write Entities to Silver

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

# Create entities DataFrame
entities_df = spark.createDataFrame(all_entities, schema=ENTITY_SCHEMA)
entities_df = entities_df.withColumn("_ingested_at", current_timestamp())

# Write entities
entities_path = f"{SILVER_PATH}/entities"

try:
    delta_table = DeltaTable.forPath(spark, entities_path)
    delta_table.alias("target").merge(
        entities_df.alias("source"),
        "target.entity_id = source.entity_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print(f"Merged {entities_df.count()} entities")
except Exception:
    entities_df.write.format("delta").mode("overwrite").save(entities_path)
    print(f"Created entities table with {entities_df.count()} entities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write Entity Registry

# COMMAND ----------

# Convert registry to list
registry_data = []
for canonical_id, data in entity_registry.items():
    registry_data.append({
        "canonical_id": canonical_id,
        "canonical_name": data["canonical_name"],
        "label": data["label"],
        "mention_count": data["mention_count"],
        "document_count": len(data["documents"]),
        "variants": list(data["variants"]),
    })

# Create DataFrame
registry_df = spark.createDataFrame(registry_data, schema=ENTITY_REGISTRY_SCHEMA)

# Write registry
registry_path = f"{SILVER_PATH}/entity_registry"
registry_df.write.format("delta").mode("overwrite").save(registry_path)
print(f"Created entity registry with {registry_df.count()} unique entities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Analyze Entities

# COMMAND ----------

# Read back
entities_silver = spark.read.format("delta").load(entities_path)
registry_silver = spark.read.format("delta").load(registry_path)

print(f"Total entity mentions: {entities_silver.count()}")
print(f"Unique entities: {registry_silver.count()}")

# COMMAND ----------

# Entities by type
print("\nEntity distribution by type:")
entities_silver.groupBy("label").count().orderBy(col("count").desc()).display()

# COMMAND ----------

# Top entities by mention count
print("\nTop 20 entities by mention count:")
registry_silver.orderBy(col("mention_count").desc()).limit(20).display()

# COMMAND ----------

# Entities per document
from pyspark.sql.functions import countDistinct

print("\nEntity diversity per document:")
entities_silver.groupBy("parent_document_id").agg(
    countDistinct("canonical_id" if "canonical_id" in entities_silver.columns else "normalized_text").alias("unique_entities")
).agg(
    avg("unique_entities").alias("avg_entities_per_doc"),
    min("unique_entities").alias("min_entities"),
    max("unique_entities").alias("max_entities"),
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Build Domain Glossary

# COMMAND ----------

# Create glossary from high-frequency entities
MIN_FREQUENCY = 3

glossary = registry_silver.filter(col("mention_count") >= MIN_FREQUENCY).orderBy(col("mention_count").desc())

print(f"Domain glossary entries (frequency >= {MIN_FREQUENCY}): {glossary.count()}")

# Save glossary
glossary_path = f"{SILVER_PATH}/domain_glossary"
glossary.write.format("delta").mode("overwrite").save(glossary_path)

display(glossary.limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary

# COMMAND ----------

total_mentions = entities_silver.count()
unique_entities = registry_silver.count()
glossary_entries = glossary.count()

# Entity type distribution
type_dist = entities_silver.groupBy("label").count().collect()
type_str = ", ".join([f"{r['label']}: {r['count']}" for r in sorted(type_dist, key=lambda x: -x['count'])[:5]])

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║            NER EXTRACTION COMPLETE                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Entity Mentions   : {total_mentions:<35} ║
║  Unique Entities         : {unique_entities:<35} ║
║  Domain Glossary Entries : {glossary_entries:<35} ║
╠══════════════════════════════════════════════════════════════════╣
║  TOP ENTITY TYPES:                                               ║
║  {type_str:<63} ║
╠══════════════════════════════════════════════════════════════════╣
║  OUTPUT TABLES:                                                  ║
║  • {SILVER_PATH}/entities - All entity mentions                 ║
║  • {SILVER_PATH}/entity_registry - Deduplicated entities        ║
║  • {SILVER_PATH}/domain_glossary - High-frequency terms         ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS:                                                     ║
║  1. Run 03_anonymization.py for PII removal                      ║
║  2. Run 04_summarization.py for chunk summaries                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
