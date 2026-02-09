# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Language Detection Pipeline (EN/NL)
# MAGIC
# MAGIC **Phase 1: Setup & Ingestion | Week 2**
# MAGIC
# MAGIC This notebook detects the language of documents and emails for proper NLP model selection.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Detect language (English / Dutch) for all Bronze content
# MAGIC - Handle mixed-language documents
# MAGIC - Tag content for downstream NLP pipeline
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC BRONZE (emails, documents) → BRONZE (+ language metadata)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langdetect pyspark delta-spark

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

try:
    STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")
    CONTAINER = dbutils.widgets.get("container")
except Exception:
    STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
    CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
BRONZE_PATH = f"{BASE_PATH}/bronze"

print(f"BRONZE_PATH: {BRONZE_PATH}")

# Supported languages
SUPPORTED_LANGUAGES = ["en", "nl"]  # English, Dutch
DEFAULT_LANGUAGE = "en"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Language Detection Function

# COMMAND ----------

from langdetect import detect, detect_langs, LangDetectException
from typing import Tuple, List, Optional
import re

def clean_text_for_detection(text: str) -> str:
    """Clean text for better language detection."""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very short lines (headers, signatures)
    lines = [l for l in text.split('\n') if len(l.strip()) > 20]
    return ' '.join(lines)

def detect_language(text: str, min_confidence: float = 0.7) -> Tuple[str, float]:
    """
    Detect language of text.

    Args:
        text: Input text
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (language_code, confidence)
    """
    if not text or len(text.strip()) < 20:
        return DEFAULT_LANGUAGE, 0.0

    try:
        cleaned = clean_text_for_detection(text)
        if len(cleaned) < 20:
            return DEFAULT_LANGUAGE, 0.0

        # Get language probabilities
        langs = detect_langs(cleaned[:5000])  # Limit for performance

        # Find best supported language
        for lang in langs:
            if lang.lang in SUPPORTED_LANGUAGES:
                if lang.prob >= min_confidence:
                    return lang.lang, lang.prob

        # Fallback: use top detected if confidence is high
        if langs and langs[0].prob >= 0.8:
            return langs[0].lang, langs[0].prob

        return DEFAULT_LANGUAGE, 0.0

    except LangDetectException:
        return DEFAULT_LANGUAGE, 0.0

def detect_languages_in_text(text: str) -> List[Tuple[str, float]]:
    """
    Detect multiple languages in text (for mixed-language documents).

    Returns list of (language, percentage) tuples.
    """
    if not text or len(text.strip()) < 50:
        return [(DEFAULT_LANGUAGE, 1.0)]

    try:
        langs = detect_langs(text[:10000])
        return [(l.lang, l.prob) for l in langs if l.lang in SUPPORTED_LANGUAGES or l.prob > 0.1]
    except LangDetectException:
        return [(DEFAULT_LANGUAGE, 1.0)]

# Test
test_en = "This is a test document written in English about machine learning."
test_nl = "Dit is een testdocument geschreven in het Nederlands over machine learning."

print(f"English test: {detect_language(test_en)}")
print(f"Dutch test: {detect_language(test_nl)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create UDF for Spark

# COMMAND ----------

from pyspark.sql.functions import udf, col, struct
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType

# Schema for language detection result
LANG_RESULT_SCHEMA = StructType([
    StructField("detected_language", StringType()),
    StructField("language_confidence", FloatType()),
])

@udf(returnType=LANG_RESULT_SCHEMA)
def detect_language_udf(text):
    """UDF for language detection."""
    if not text:
        return {"detected_language": DEFAULT_LANGUAGE, "language_confidence": 0.0}

    lang, conf = detect_language(text)
    return {"detected_language": lang, "language_confidence": float(conf)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Process Emails

# COMMAND ----------

from delta.tables import DeltaTable

# Read emails
emails_path = f"{BRONZE_PATH}/emails"
try:
    emails_df = spark.read.format("delta").load(emails_path)
    print(f"Loaded {emails_df.count()} emails")
except Exception as e:
    print(f"No emails table found: {e}")
    emails_df = None

# COMMAND ----------

if emails_df is not None:
    # Add language detection
    emails_with_lang = emails_df.withColumn(
        "lang_result",
        detect_language_udf(col("body_text"))
    ).select(
        "*",
        col("lang_result.detected_language").alias("detected_language"),
        col("lang_result.language_confidence").alias("language_confidence"),
    ).drop("lang_result")

    # Language distribution
    print("Email Language Distribution:")
    emails_with_lang.groupBy("detected_language").count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Update Emails with Language

# COMMAND ----------

if emails_df is not None:
    # Write back to Delta with language columns
    emails_with_lang.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(emails_path)
    print(f"Updated emails table with language detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Process Documents

# COMMAND ----------

# Read documents
docs_path = f"{BRONZE_PATH}/documents"
try:
    docs_df = spark.read.format("delta").load(docs_path)
    print(f"Loaded {docs_df.count()} documents")
except Exception as e:
    print(f"No documents table found: {e}")
    docs_df = None

# COMMAND ----------

if docs_df is not None:
    # Add language detection
    docs_with_lang = docs_df.withColumn(
        "lang_result",
        detect_language_udf(col("content"))
    ).select(
        "*",
        col("lang_result.detected_language").alias("detected_language"),
        col("lang_result.language_confidence").alias("language_confidence"),
    ).drop("lang_result")

    # Language distribution
    print("Document Language Distribution:")
    docs_with_lang.groupBy("detected_language").count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Update Documents with Language

# COMMAND ----------

if docs_df is not None:
    # Write back
    docs_with_lang.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(docs_path)
    print(f"Updated documents table with language detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Language Summary

# COMMAND ----------

from pyspark.sql.functions import count, avg, when

# Combined summary
summary_data = []

if emails_df is not None:
    emails_updated = spark.read.format("delta").load(emails_path)
    email_stats = emails_updated.groupBy("detected_language").agg(
        count("*").alias("count"),
        avg("language_confidence").alias("avg_confidence")
    ).collect()

    for row in email_stats:
        summary_data.append({
            "source": "emails",
            "language": row["detected_language"],
            "count": row["count"],
            "avg_confidence": row["avg_confidence"],
        })

if docs_df is not None:
    docs_updated = spark.read.format("delta").load(docs_path)
    doc_stats = docs_updated.groupBy("detected_language").agg(
        count("*").alias("count"),
        avg("language_confidence").alias("avg_confidence")
    ).collect()

    for row in doc_stats:
        summary_data.append({
            "source": "documents",
            "language": row["detected_language"],
            "count": row["count"],
            "avg_confidence": row["avg_confidence"],
        })

# Display summary
if summary_data:
    summary_df = spark.createDataFrame(summary_data)
    display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Low Confidence Items

# COMMAND ----------

# Identify items with low language confidence for review
LOW_CONFIDENCE_THRESHOLD = 0.5

if emails_df is not None:
    low_conf_emails = spark.read.format("delta").load(emails_path).filter(
        col("language_confidence") < LOW_CONFIDENCE_THRESHOLD
    )
    print(f"Emails with low language confidence: {low_conf_emails.count()}")

if docs_df is not None:
    low_conf_docs = spark.read.format("delta").load(docs_path).filter(
        col("language_confidence") < LOW_CONFIDENCE_THRESHOLD
    )
    print(f"Documents with low language confidence: {low_conf_docs.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary & Next Steps

# COMMAND ----------

total_emails = spark.read.format("delta").load(emails_path).count() if emails_df is not None else 0
total_docs = spark.read.format("delta").load(docs_path).count() if docs_df is not None else 0

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║            LANGUAGE DETECTION COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Emails Processed    : {total_emails:<35} ║
║  Total Documents Processed : {total_docs:<35} ║
║  Supported Languages       : English (en), Dutch (nl)            ║
╠══════════════════════════════════════════════════════════════════╣
║  NEW COLUMNS ADDED:                                              ║
║  • detected_language   - ISO language code (en/nl)               ║
║  • language_confidence - Detection confidence (0.0-1.0)          ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS (Phase 2):                                           ║
║  1. Run chunking pipeline (semantic chunking)                    ║
║  2. Run NER extraction (spaCy en/nl models)                      ║
║  3. Run anonymization pipeline (Presidio)                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1 Complete!
# MAGIC
# MAGIC ### Milestone M1 Deliverables:
# MAGIC - [x] Azure environment configured
# MAGIC - [x] Email ingestion pipeline (PST/MSG)
# MAGIC - [x] Document ingestion pipeline (PDF, DOCX, XLSX, PPTX)
# MAGIC - [x] Language detection (EN/NL)
# MAGIC - [x] Bronze zone populated with metadata
# MAGIC
# MAGIC ### Ready for Week 2 Presentation Demo:
# MAGIC - Show ingestion pipeline processing sample data
# MAGIC - Demonstrate language detection results
# MAGIC - Display Bronze zone statistics
