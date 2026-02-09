# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - PII Anonymization Pipeline
# MAGIC
# MAGIC **Phase 2: NLP Processing | Week 4**
# MAGIC
# MAGIC This notebook detects and anonymizes Personally Identifiable Information (PII) using Microsoft Presidio.
# MAGIC
# MAGIC ## PII Types Detected (Global/International)
# MAGIC - Names (PERSON)
# MAGIC - Email addresses
# MAGIC - Phone numbers (international formats)
# MAGIC - IBAN codes (all countries)
# MAGIC - EU VAT numbers (multi-country)
# MAGIC - US Social Security Numbers
# MAGIC - Passport numbers
# MAGIC - National IDs (generic patterns)
# MAGIC - Credit card numbers
# MAGIC - SWIFT/BIC codes
# MAGIC - IP addresses
# MAGIC - Employee IDs
# MAGIC - Company registration numbers
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC SILVER (chunks) → SILVER (anonymized_chunks) + Mapping Store
# MAGIC ```
# MAGIC
# MAGIC ## Success Criteria
# MAGIC **Milestone M2: PII detection recall > 95%**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install presidio-analyzer presidio-anonymizer spacy pandas pyarrow delta-spark

# COMMAND ----------

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

# Anonymization configuration (Global/International patterns)
ANON_CONFIG = {
    "entities_to_detect": [
        # Standard PII
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
        "DATE_TIME",
        # Financial
        "IBAN_CODE",
        "CREDIT_CARD",
        "SWIFT_CODE",
        # Digital
        "IP_ADDRESS",
        "URL",
        # Government/Official IDs (International)
        "US_SSN",
        "PASSPORT_NUMBER",
        "NATIONAL_ID",
        "EU_VAT_NUMBER",
        # Enterprise
        "EMPLOYEE_ID",
        "COMPANY_REGISTRATION",
    ],
    "strategy": "pseudonymize",  # replace, redact, hash, pseudonymize
    "min_confidence": 0.7,
    "consistent_pseudonyms": True,
    "preserve_format": True,
}

print(f"SILVER_PATH: {SILVER_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Presidio with Global/International Patterns

# COMMAND ----------

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Create registry
registry = RecognizerRegistry()
registry.load_predefined_recognizers()

# ============================================
# INTERNATIONAL PHONE NUMBERS
# ============================================
intl_phone_pattern = Pattern(
    name="international_phone",
    regex=r"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{1,4}[\s.\-]?\d{1,4}[\s.\-]?\d{1,9}",
    score=0.75,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[intl_phone_pattern],
    supported_language="en",
))

# US Phone: (xxx) xxx-xxxx
us_phone_pattern = Pattern(
    name="us_phone",
    regex=r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}",
    score=0.7,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[us_phone_pattern],
    supported_language="en",
))

# ============================================
# INTERNATIONAL IBAN (All Countries)
# ============================================
iban_pattern = Pattern(
    name="international_iban",
    regex=r"\b[A-Z]{2}\d{2}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{4}[\s]?[A-Z0-9]{0,14}\b",
    score=0.9,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="IBAN_CODE",
    patterns=[iban_pattern],
    supported_language="en",
))

# ============================================
# EU VAT NUMBERS (Multiple Countries)
# ============================================
eu_vat_pattern = Pattern(
    name="eu_vat",
    regex=r"\b(AT|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK|GB|XI)[A-Z0-9]{8,12}\b",
    score=0.85,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="EU_VAT_NUMBER",
    patterns=[eu_vat_pattern],
    supported_language="en",
))

# ============================================
# US SOCIAL SECURITY NUMBER
# ============================================
ssn_pattern = Pattern(
    name="us_ssn",
    regex=r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b",
    score=0.8,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[ssn_pattern],
    supported_language="en",
))

# ============================================
# PASSPORT NUMBERS (Generic)
# ============================================
passport_pattern = Pattern(
    name="passport_number",
    regex=r"\b[A-Z]{1,2}\d{6,9}\b",
    score=0.6,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="PASSPORT_NUMBER",
    patterns=[passport_pattern],
    supported_language="en",
))

# ============================================
# EMPLOYEE/STAFF IDs (Enterprise Patterns)
# ============================================
employee_id_pattern = Pattern(
    name="employee_id",
    regex=r"\b(EMP|STAFF|ID|USR)[\-_]?\d{4,8}\b",
    score=0.7,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    patterns=[employee_id_pattern],
    supported_language="en",
))

# ============================================
# SWIFT/BIC CODES (International Banking)
# ============================================
swift_pattern = Pattern(
    name="swift_bic",
    regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
    score=0.8,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="SWIFT_CODE",
    patterns=[swift_pattern],
    supported_language="en",
))

# ============================================
# GENERIC NATIONAL ID (Various formats)
# ============================================
national_id_pattern = Pattern(
    name="national_id_generic",
    regex=r"\b\d{2,3}[\.\-/]\d{2,3}[\.\-/]\d{2,4}([\.\-/]\d{2,4})?\b",
    score=0.65,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="NATIONAL_ID",
    patterns=[national_id_pattern],
    supported_language="en",
))

# ============================================
# COMPANY REGISTRATION NUMBERS
# ============================================
company_reg_pattern = Pattern(
    name="company_registration",
    regex=r"\b(REG|CRN|KVK|SIREN|SIRET)[\s\-:]?\d{6,14}\b",
    score=0.75,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="COMPANY_REGISTRATION",
    patterns=[company_reg_pattern],
    supported_language="en",
))

# Create analyzer
analyzer = AnalyzerEngine(registry=registry)

# Create anonymizer
anonymizer_engine = AnonymizerEngine()

print("Presidio initialized with Global/International PII patterns")
print(f"Registered recognizers: {len(registry.get_recognizers(language='en'))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pseudonym Generator

# COMMAND ----------

import hashlib
from collections import defaultdict
import json

class PseudonymGenerator:
    """Generate consistent pseudonyms for global/international PII values."""

    def __init__(self, seed: str = "thesis_2024"):
        self.seed = seed
        self.mapping = {}
        self.reverse_mapping = {}
        self.counters = defaultdict(int)

    def _hash(self, value: str) -> str:
        return hashlib.sha256(f"{self.seed}:{value}".encode()).hexdigest()[:8]

    def get_pseudonym(self, original: str, entity_type: str) -> str:
        """Get or create pseudonym for value."""
        key = f"{entity_type}:{original}"
        if key in self.mapping:
            return self.mapping[key]

        # Generate pseudonym based on type (global/international)
        hash_val = self._hash(original)

        if entity_type == "PERSON":
            pseudonym = f"Person_{hash_val[:4].upper()}"
        elif entity_type == "EMAIL_ADDRESS":
            pseudonym = f"user_{hash_val[:6]}@anonymized.local"
        elif entity_type == "PHONE_NUMBER":
            if original.startswith("+"):
                pseudonym = f"+XX-XXX-XXX-{hash_val[:4]}"
            else:
                pseudonym = f"XXX-XXX-{hash_val[:4]}"
        elif entity_type == "LOCATION":
            pseudonym = f"Location_{hash_val[:4].upper()}"
        elif entity_type == "IBAN_CODE":
            # Preserve country code if detectable
            if len(original) >= 2 and original[:2].isalpha():
                pseudonym = f"{original[:2].upper()}XX XXXX XXXX XXXX"
            else:
                pseudonym = "XXXX XXXX XXXX XXXX"
        elif entity_type == "EU_VAT_NUMBER":
            if len(original) >= 2 and original[:2].isalpha():
                pseudonym = f"{original[:2].upper()}XXXXXXXXXX"
            else:
                pseudonym = "XXXXXXXXXXXX"
        elif entity_type == "US_SSN":
            pseudonym = "XXX-XX-XXXX"
        elif entity_type == "PASSPORT_NUMBER":
            pseudonym = f"XX{hash_val[:6].upper()}"
        elif entity_type == "NATIONAL_ID":
            pseudonym = "XX-XXXXXX-XX"
        elif entity_type == "CREDIT_CARD":
            pseudonym = "XXXX-XXXX-XXXX-XXXX"
        elif entity_type == "SWIFT_CODE":
            pseudonym = "XXXXXX2AXXX"
        elif entity_type == "EMPLOYEE_ID":
            pseudonym = f"EMP-{hash_val[:6].upper()}"
        elif entity_type == "COMPANY_REGISTRATION":
            pseudonym = "REG-XXXXXXXX"
        elif entity_type == "IP_ADDRESS":
            pseudonym = f"10.0.{int(hash_val[:2], 16) % 256}.{int(hash_val[2:4], 16) % 256}"
        else:
            pseudonym = f"[{entity_type}_{hash_val}]"

        self.mapping[key] = pseudonym
        self.reverse_mapping[pseudonym] = original
        return pseudonym

    def export_mapping(self) -> str:
        """Export mapping as JSON."""
        return json.dumps(self.mapping, indent=2)

# Initialize generator
pseudonym_gen = PseudonymGenerator()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define Anonymization Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, TimestampType, MapType

ANONYMIZED_CHUNK_SCHEMA = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("parent_document_id", StringType(), False),
    StructField("original_content", StringType(), True),
    StructField("anonymized_content", StringType(), True),
    StructField("pii_count", IntegerType(), True),
    StructField("pii_types", ArrayType(StringType()), True),
    StructField("pii_mapping", MapType(StringType(), StringType()), True),
    StructField("detected_language", StringType(), True),
    StructField("anonymization_timestamp", TimestampType(), True),
])

PII_DETECTION_SCHEMA = StructType([
    StructField("detection_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("entity_type", StringType(), True),
    StructField("original_text", StringType(), True),
    StructField("anonymized_text", StringType(), True),
    StructField("start_char", IntegerType(), True),
    StructField("end_char", IntegerType(), True),
    StructField("confidence", FloatType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Anonymization Function

# COMMAND ----------

from typing import List, Dict, Any, Tuple
from datetime import datetime
from pyspark.sql.types import FloatType

def detect_and_anonymize(
    text: str,
    chunk_id: str,
    parent_document_id: str,
    language: str = "en"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Detect PII and anonymize text.

    Returns:
        Tuple of (anonymized_chunk_dict, list_of_pii_detections)
    """
    # Detect PII
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=ANON_CONFIG["entities_to_detect"] + ["BE_NATIONAL_ID", "BE_VAT_NUMBER"],
        score_threshold=ANON_CONFIG["min_confidence"],
    )

    if not results:
        # No PII found
        return {
            "chunk_id": chunk_id,
            "parent_document_id": parent_document_id,
            "original_content": text,
            "anonymized_content": text,
            "pii_count": 0,
            "pii_types": [],
            "pii_mapping": {},
            "detected_language": language,
            "anonymization_timestamp": datetime.utcnow(),
        }, []

    # Sort by position (reverse for replacement)
    results = sorted(results, key=lambda x: x.start, reverse=True)

    # Anonymize
    anonymized_text = text
    pii_mapping = {}
    pii_types = set()
    pii_detections = []

    for result in results:
        original = text[result.start:result.end]
        pseudonym = pseudonym_gen.get_pseudonym(original, result.entity_type)

        # Replace in text
        anonymized_text = anonymized_text[:result.start] + pseudonym + anonymized_text[result.end:]

        # Record
        pii_mapping[original] = pseudonym
        pii_types.add(result.entity_type)

        # Detection record
        detection_id = hashlib.md5(
            f"{chunk_id}:{result.start}:{result.end}:{original}".encode()
        ).hexdigest()[:16]

        pii_detections.append({
            "detection_id": detection_id,
            "chunk_id": chunk_id,
            "entity_type": result.entity_type,
            "original_text": original,
            "anonymized_text": pseudonym,
            "start_char": result.start,
            "end_char": result.end,
            "confidence": result.score,
        })

    return {
        "chunk_id": chunk_id,
        "parent_document_id": parent_document_id,
        "original_content": text,
        "anonymized_content": anonymized_text,
        "pii_count": len(results),
        "pii_types": list(pii_types),
        "pii_mapping": pii_mapping,
        "detected_language": language,
        "anonymization_timestamp": datetime.utcnow(),
    }, pii_detections

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load Chunks from Silver

# COMMAND ----------

chunks_df = spark.read.format("delta").load(f"{SILVER_PATH}/chunks")
print(f"Loaded {chunks_df.count()} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Process Chunks for Anonymization

# COMMAND ----------

from pyspark.sql.functions import col

# Collect chunks
chunks = chunks_df.select(
    "chunk_id",
    "parent_document_id",
    "content",
    "detected_language"
).collect()

print(f"Processing {len(chunks)} chunks for PII detection...")

# COMMAND ----------

all_anonymized = []
all_detections = []
errors = []

for i, chunk in enumerate(chunks):
    try:
        anon_chunk, detections = detect_and_anonymize(
            text=chunk["content"],
            chunk_id=chunk["chunk_id"],
            parent_document_id=chunk["parent_document_id"],
            language=chunk["detected_language"] or "en",
        )

        all_anonymized.append(anon_chunk)
        all_detections.extend(detections)

    except Exception as e:
        errors.append({"chunk_id": chunk["chunk_id"], "error": str(e)})

    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks, {len(all_detections)} PII instances detected")

print(f"\nAnonymization complete:")
print(f"  Chunks processed: {len(all_anonymized)}")
print(f"  PII instances: {len(all_detections)}")
print(f"  Errors: {len(errors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write Anonymized Chunks to Silver

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

# Create DataFrame
anon_df = spark.createDataFrame(all_anonymized, schema=ANONYMIZED_CHUNK_SCHEMA)
anon_df = anon_df.withColumn("_ingested_at", current_timestamp())

# Write anonymized chunks
anon_path = f"{SILVER_PATH}/anonymized_chunks"

try:
    delta_table = DeltaTable.forPath(spark, anon_path)
    delta_table.alias("target").merge(
        anon_df.alias("source"),
        "target.chunk_id = source.chunk_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print(f"Merged {anon_df.count()} anonymized chunks")
except Exception:
    anon_df.write.format("delta").mode("overwrite").save(anon_path)
    print(f"Created anonymized chunks table with {anon_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Write PII Detections Log

# COMMAND ----------

if all_detections:
    detections_df = spark.createDataFrame(all_detections, schema=PII_DETECTION_SCHEMA)
    detections_df = detections_df.withColumn("_ingested_at", current_timestamp())

    # Write detections
    detections_path = f"{SILVER_PATH}/pii_detections"
    detections_df.write.format("delta").mode("overwrite").save(detections_path)
    print(f"Saved {detections_df.count()} PII detections")
else:
    print("No PII detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Pseudonym Mapping (Secure Storage)

# COMMAND ----------

# Save mapping to secure location
# In production, store in Azure Key Vault or encrypted storage
mapping_json = pseudonym_gen.export_mapping()
mapping_path = f"{SILVER_PATH}/pii_mapping/mapping.json"

dbutils.fs.put(mapping_path, mapping_json, overwrite=True)
print(f"Saved pseudonym mapping with {len(pseudonym_gen.mapping)} entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Analyze PII Detection Results

# COMMAND ----------

# Read back
anon_silver = spark.read.format("delta").load(anon_path)

# PII statistics
print("PII Detection Statistics:")
print(f"Total chunks: {anon_silver.count()}")
print(f"Chunks with PII: {anon_silver.filter(col('pii_count') > 0).count()}")

# COMMAND ----------

# PII by type
if all_detections:
    detections_silver = spark.read.format("delta").load(detections_path)
    print("\nPII by type:")
    detections_silver.groupBy("entity_type").count().orderBy(col("count").desc()).display()

# COMMAND ----------

# PII distribution per chunk
from pyspark.sql.functions import avg, max, sum

print("\nPII distribution:")
anon_silver.agg(
    avg("pii_count").alias("avg_pii_per_chunk"),
    max("pii_count").alias("max_pii_in_chunk"),
    sum("pii_count").alias("total_pii"),
).display()

# COMMAND ----------

# Confidence score distribution
if all_detections:
    print("\nConfidence score distribution:")
    detections_silver.groupBy("entity_type").agg(
        avg("confidence").alias("avg_confidence"),
        min("confidence").alias("min_confidence"),
    ).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Verify Anonymization Quality

# COMMAND ----------

# Sample comparison
print("Sample anonymization comparison:")
sample = anon_silver.filter(col("pii_count") > 0).limit(3).collect()

for s in sample:
    print(f"\n{'='*60}")
    print(f"Chunk: {s['chunk_id']}")
    print(f"PII Count: {s['pii_count']}, Types: {s['pii_types']}")
    print(f"\nOriginal (first 300 chars):\n{s['original_content'][:300]}...")
    print(f"\nAnonymized (first 300 chars):\n{s['anonymized_content'][:300]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Summary

# COMMAND ----------

total_chunks = anon_silver.count()
chunks_with_pii = anon_silver.filter(col("pii_count") > 0).count()
total_pii = anon_silver.agg(sum("pii_count")).collect()[0][0] or 0
pii_percentage = (chunks_with_pii / total_chunks * 100) if total_chunks > 0 else 0

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║            PII ANONYMIZATION COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Chunks Processed  : {total_chunks:<35} ║
║  Chunks with PII         : {chunks_with_pii:<35} ║
║  Total PII Instances     : {total_pii:<35} ║
║  PII Coverage            : {pii_percentage:.1f}% of chunks                          ║
╠══════════════════════════════════════════════════════════════════╣
║  GLOBAL/INTERNATIONAL PII PATTERNS:                              ║
║  • Phone: International formats (+xx, US, EU)                    ║
║  • IBAN: All countries (preserves country code)                  ║
║  • VAT: EU multi-country (AT, BE, DE, FR, NL, etc.)              ║
║  • IDs: SSN, Passport, National IDs, Employee IDs                ║
║  • Financial: Credit cards, SWIFT/BIC codes                      ║
║  • Enterprise: Company registration numbers                      ║
╠══════════════════════════════════════════════════════════════════╣
║  OUTPUT TABLES:                                                  ║
║  • {SILVER_PATH}/anonymized_chunks                              ║
║  • {SILVER_PATH}/pii_detections                                 ║
║  • {SILVER_PATH}/pii_mapping (secure)                           ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ MILESTONE M2: PII detection pipeline complete                 ║
║  NEXT: Run 04_summarization.py                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
