# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4.1: GraphRAG Entity Extraction
# MAGIC
# MAGIC Extract entities and relationships from processed chunks using GPT-4o with Pydantic structured outputs.
# MAGIC
# MAGIC **Week 6 - Knowledge Graph Construction**
# MAGIC
# MAGIC ## Features
# MAGIC - LLM-based entity extraction (PERSON, ORGANIZATION, PROJECT, TECHNOLOGY, DOCUMENT, EVENT, LOCATION)
# MAGIC - Relationship extraction (WORKS_ON, REPORTS_TO, MENTIONS, USES, PART_OF, etc.)
# MAGIC - Entity normalization and deduplication
# MAGIC - Batch processing with progress tracking
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai pydantic tenacity delta-spark

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
from pyspark.sql.functions import col, lit, udf, struct, collect_list, count
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, FloatType, IntegerType

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from graphrag.entity_extraction import (
    GraphRAGEntityExtractor,
    EntityNormalizer,
    RelationshipProcessor,
    ExtractionConfig,
    ExtractionResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
# Load from Databricks secrets
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="azure-openai", key="endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="azure-openai", key="api-key")

# Validate
assert AZURE_OPENAI_ENDPOINT, "Azure OpenAI endpoint not configured"
assert AZURE_OPENAI_KEY, "Azure OpenAI key not configured"

print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT[:50]}...")

# COMMAND ----------

# DBTITLE 1,Configure Delta Lake Paths
# Delta Lake paths
BRONZE_PATH = "/mnt/datalake/bronze"
SILVER_PATH = "/mnt/datalake/silver"
GOLD_PATH = "/mnt/datalake/gold"

# Input: Processed chunks from Phase 2
CHUNKS_TABLE = f"{SILVER_PATH}/chunks"
CHUNKS_ANONYMIZED_TABLE = f"{SILVER_PATH}/chunks_anonymized"

# Output: Extracted entities and relationships
ENTITIES_RAW_TABLE = f"{SILVER_PATH}/entities_raw"
RELATIONSHIPS_RAW_TABLE = f"{SILVER_PATH}/relationships_raw"
ENTITIES_NORMALIZED_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"

print("Delta Lake paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Entity Extractor

# COMMAND ----------

# DBTITLE 1,Configure Extraction Settings
# Entity extraction configuration
extraction_config = ExtractionConfig(
    # Entity types to extract
    entity_types=[
        "PERSON", "ORGANIZATION", "PROJECT", "TECHNOLOGY",
        "DOCUMENT", "EVENT", "LOCATION"
    ],
    # Relationship types
    relationship_types=[
        "WORKS_ON", "REPORTS_TO", "MENTIONS", "USES",
        "PART_OF", "AUTHORED", "ATTENDED", "LOCATED_IN"
    ],
    # Model settings
    model_deployment="gpt-4o",
    temperature=0.0,  # Deterministic for consistency
    max_tokens=2000,
    # Extraction limits
    max_entities_per_chunk=20,
    max_relationships_per_chunk=30,
    max_chunk_length=8000
)

print("Extraction configuration:")
print(f"  Entity types: {extraction_config.entity_types}")
print(f"  Relationship types: {extraction_config.relationship_types}")
print(f"  Model: {extraction_config.model_deployment}")

# COMMAND ----------

# DBTITLE 1,Initialize Extractor
# Initialize entity extractor
extractor = GraphRAGEntityExtractor(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    config=extraction_config
)

# Initialize normalizer and relationship processor
entity_normalizer = EntityNormalizer()
relationship_processor = RelationshipProcessor(entity_normalizer)

print("Entity extractor initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Processed Chunks

# COMMAND ----------

# DBTITLE 1,Load Chunks from Silver Layer
# Load anonymized chunks (preferred) or regular chunks
try:
    chunks_df = spark.read.format("delta").load(CHUNKS_ANONYMIZED_TABLE)
    print(f"Loaded anonymized chunks from {CHUNKS_ANONYMIZED_TABLE}")
except:
    chunks_df = spark.read.format("delta").load(CHUNKS_TABLE)
    print(f"Loaded chunks from {CHUNKS_TABLE}")

# Display schema and count
print(f"\nTotal chunks: {chunks_df.count()}")
chunks_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Sample Chunks for Testing
# For development, sample a subset
SAMPLE_SIZE = 100  # Adjust for full run
sample_chunks = chunks_df.limit(SAMPLE_SIZE).collect()

print(f"Sampled {len(sample_chunks)} chunks for entity extraction")

# Display sample
display(chunks_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entity Extraction Pipeline

# COMMAND ----------

# DBTITLE 1,Define Extraction Function
def extract_from_chunk(chunk_row) -> Dict[str, Any]:
    """
    Extract entities and relationships from a single chunk.

    Args:
        chunk_row: Spark Row with chunk data

    Returns:
        Dictionary with extraction results
    """
    try:
        # Get chunk text and metadata
        chunk_id = chunk_row.chunk_id
        text = chunk_row.text if hasattr(chunk_row, 'text') else chunk_row.content
        source_file = chunk_row.source_file if hasattr(chunk_row, 'source_file') else None

        # Extract entities and relationships
        result, metadata = extractor.extract(
            text=text,
            chunk_id=chunk_id,
            metadata={"source_file": source_file}
        )

        # Convert to serializable format
        entities = [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "chunk_id": chunk_id,
                "source_file": source_file
            }
            for e in result.entities
        ]

        relationships = [
            {
                "source": r.source,
                "target": r.target,
                "relationship": r.relationship,
                "description": r.description,
                "strength": r.strength,
                "chunk_id": chunk_id
            }
            for r in result.relationships
        ]

        return {
            "chunk_id": chunk_id,
            "entities": entities,
            "relationships": relationships,
            "num_entities": len(entities),
            "num_relationships": len(relationships),
            "success": True,
            "error": None
        }

    except Exception as e:
        logger.error(f"Extraction failed for chunk {chunk_row.chunk_id}: {e}")
        return {
            "chunk_id": chunk_row.chunk_id,
            "entities": [],
            "relationships": [],
            "num_entities": 0,
            "num_relationships": 0,
            "success": False,
            "error": str(e)
        }

# COMMAND ----------

# DBTITLE 1,Run Batch Extraction
from tqdm import tqdm

# Process chunks with progress tracking
extraction_results = []
total_entities = 0
total_relationships = 0
errors = 0

print(f"Starting entity extraction for {len(sample_chunks)} chunks...")
print("-" * 60)

for i, chunk in enumerate(tqdm(sample_chunks, desc="Extracting entities")):
    result = extract_from_chunk(chunk)
    extraction_results.append(result)

    total_entities += result["num_entities"]
    total_relationships += result["num_relationships"]

    if not result["success"]:
        errors += 1

    # Progress update every 10 chunks
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(sample_chunks)} chunks | "
              f"Entities: {total_entities} | Relationships: {total_relationships}")

print("-" * 60)
print(f"\nExtraction complete!")
print(f"  Total entities extracted: {total_entities}")
print(f"  Total relationships extracted: {total_relationships}")
print(f"  Errors: {errors}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Entity Normalization and Deduplication

# COMMAND ----------

# DBTITLE 1,Normalize Entities
# Collect all raw entities
all_raw_entities = []
for result in extraction_results:
    all_raw_entities.extend(result["entities"])

print(f"Raw entities before normalization: {len(all_raw_entities)}")

# Normalize and deduplicate
for entity_dict in all_raw_entities:
    from graphrag.entity_extraction import ExtractedEntity

    entity = ExtractedEntity(
        name=entity_dict["name"],
        type=entity_dict["type"],
        description=entity_dict["description"]
    )

    entity_normalizer.register_entity(
        entity=entity,
        chunk_id=entity_dict["chunk_id"],
        source_file=entity_dict.get("source_file")
    )

# Get normalized entities
normalized_entities = entity_normalizer.get_all_entities()
print(f"Normalized entities after deduplication: {len(normalized_entities)}")

# Statistics
entity_stats = entity_normalizer.get_statistics()
print(f"\nEntity statistics:")
print(f"  Total unique entities: {entity_stats['total_entities']}")
print(f"  Average mentions per entity: {entity_stats['avg_mentions']:.2f}")
print(f"  Max mentions: {entity_stats['max_mentions']}")
print(f"\nBy type:")
for entity_type, count in entity_stats['by_type'].items():
    print(f"  {entity_type}: {count}")

# COMMAND ----------

# DBTITLE 1,Process Relationships
# Build entity name to ID mapping
entity_name_to_id = {}
for entity in normalized_entities:
    entity_name_to_id[entity["name"]] = entity["id"]
    # Also add variants
    for variant in entity.get("variants", []):
        entity_name_to_id[variant] = entity["id"]

# Process all relationships
for result in extraction_results:
    for rel_dict in result["relationships"]:
        from graphrag.entity_extraction import ExtractedRelationship

        relationship = ExtractedRelationship(
            source=rel_dict["source"],
            target=rel_dict["target"],
            relationship=rel_dict["relationship"],
            description=rel_dict["description"],
            strength=rel_dict["strength"]
        )

        relationship_processor.add_relationship(
            relationship=relationship,
            chunk_id=rel_dict["chunk_id"],
            source_entities=entity_name_to_id,
            target_entities=entity_name_to_id
        )

# Get processed relationships
processed_relationships = relationship_processor.get_all_relationships()
print(f"Processed relationships: {len(processed_relationships)}")

# Statistics
rel_stats = relationship_processor.get_statistics()
print(f"\nRelationship statistics:")
print(f"  Total relationships: {rel_stats['total_relationships']}")
print(f"  Unique relationships: {rel_stats['unique_relationships']}")
print(f"\nBy type:")
for rel_type, count in rel_stats['by_type'].items():
    print(f"  {rel_type}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save to Delta Lake

# COMMAND ----------

# DBTITLE 1,Save Raw Extraction Results
# Create DataFrame from raw extraction results
raw_results_df = spark.createDataFrame([
    {
        "chunk_id": r["chunk_id"],
        "num_entities": r["num_entities"],
        "num_relationships": r["num_relationships"],
        "success": r["success"],
        "error": r["error"],
        "extraction_timestamp": datetime.now().isoformat()
    }
    for r in extraction_results
])

# Save extraction metadata
raw_results_df.write.format("delta").mode("overwrite").save(f"{SILVER_PATH}/extraction_metadata")
print(f"Saved extraction metadata to {SILVER_PATH}/extraction_metadata")

# COMMAND ----------

# DBTITLE 1,Save Normalized Entities
# Create entities DataFrame
entities_df = spark.createDataFrame(normalized_entities)

# Add metadata columns
entities_df = entities_df.withColumn("created_at", lit(datetime.now().isoformat()))
entities_df = entities_df.withColumn("source", lit("gpt-4o-extraction"))

# Save to Gold layer
entities_df.write.format("delta").mode("overwrite").save(ENTITIES_NORMALIZED_TABLE)
print(f"Saved {entities_df.count()} entities to {ENTITIES_NORMALIZED_TABLE}")

# Display sample
display(entities_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Save Relationships
# Create relationships DataFrame
relationships_df = spark.createDataFrame(processed_relationships)

# Add metadata columns
relationships_df = relationships_df.withColumn("created_at", lit(datetime.now().isoformat()))

# Save to Gold layer
relationships_df.write.format("delta").mode("overwrite").save(RELATIONSHIPS_TABLE)
print(f"Saved {relationships_df.count()} relationships to {RELATIONSHIPS_TABLE}")

# Display sample
display(relationships_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Extraction Quality Analysis

# COMMAND ----------

# DBTITLE 1,Entity Type Distribution
# Visualize entity type distribution
entity_type_counts = entities_df.groupBy("type").count().orderBy(col("count").desc())
display(entity_type_counts)

# COMMAND ----------

# DBTITLE 1,Relationship Type Distribution
# Visualize relationship type distribution
rel_type_counts = relationships_df.groupBy("type").count().orderBy(col("count").desc())
display(rel_type_counts)

# COMMAND ----------

# DBTITLE 1,Top Entities by Mention Count
# Top entities by mention count
top_entities = entities_df.orderBy(col("mention_count").desc()).limit(20)
display(top_entities.select("name", "type", "mention_count", "description"))

# COMMAND ----------

# DBTITLE 1,Entity Co-occurrence Analysis
# Entities that appear together frequently
from pyspark.sql.functions import explode, array

# Group relationships by source-target pairs
cooccurrence = relationships_df.groupBy("source_name", "target_name").count()
cooccurrence = cooccurrence.orderBy(col("count").desc()).limit(20)
display(cooccurrence)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Extraction Summary
print("=" * 60)
print("PHASE 4.1: ENTITY EXTRACTION COMPLETE")
print("=" * 60)

print(f"\n📊 EXTRACTION STATISTICS:")
print(f"  Chunks processed: {len(sample_chunks)}")
print(f"  Raw entities extracted: {len(all_raw_entities)}")
print(f"  Normalized entities: {len(normalized_entities)}")
print(f"  Relationships extracted: {len(processed_relationships)}")
print(f"  Extraction errors: {errors}")

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"  Entities: {ENTITIES_NORMALIZED_TABLE}")
print(f"  Relationships: {RELATIONSHIPS_TABLE}")
print(f"  Extraction metadata: {SILVER_PATH}/extraction_metadata")

print(f"\n🎯 ENTITY BREAKDOWN:")
for entity_type, count in entity_stats['by_type'].items():
    print(f"  {entity_type}: {count}")

print(f"\n🔗 RELATIONSHIP BREAKDOWN:")
for rel_type, count in rel_stats['by_type'].items():
    print(f"  {rel_type}: {count}")

print("\n" + "=" * 60)
print("NEXT: 02_knowledge_graph.py - Build Cosmos DB Graph")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Entity Types Extracted:**
# MAGIC - PERSON: People names (employees, contacts, stakeholders)
# MAGIC - ORGANIZATION: Companies, departments, teams
# MAGIC - PROJECT: Project names, initiatives
# MAGIC - TECHNOLOGY: Technologies, tools, systems
# MAGIC - DOCUMENT: Document references
# MAGIC - EVENT: Meetings, milestones, deadlines
# MAGIC - LOCATION: Offices, cities, regions
# MAGIC
# MAGIC **Relationship Types:**
# MAGIC - WORKS_ON, REPORTS_TO, MENTIONS, USES, PART_OF, AUTHORED, ATTENDED, LOCATED_IN
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Run `02_knowledge_graph.py` to populate Cosmos DB Gremlin graph
# MAGIC 2. Run `03_community_detection.py` to detect communities
# MAGIC 3. Run `04_community_summarization.py` to generate summaries
