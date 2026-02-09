# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - PST/MSG Email Ingestion Pipeline
# MAGIC
# MAGIC **Phase 1: Setup & Ingestion | Week 1-2**
# MAGIC
# MAGIC This notebook handles the ingestion of PST email archives and MSG files into the Bronze zone.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Load PST archive files (35 years of email history)
# MAGIC - Extract individual emails with metadata
# MAGIC - Preserve email threading relationships
# MAGIC - Extract and catalog attachments
# MAGIC - Store in Delta Lake Bronze zone
# MAGIC
# MAGIC ## Data Flow
# MAGIC ```
# MAGIC RAW (PST/MSG files) → BRONZE (Delta: emails, attachments, threads)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

# MAGIC %pip install extract-msg langchain langchain-community pandas pyarrow delta-spark

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Run setup notebook to get configuration
# %run ../00_setup/00_azure_environment_setup

# COMMAND ----------

# Configuration
STORAGE_ACCOUNT = dbutils.widgets.get("storage_account") if "storage_account" in [w.name for w in dbutils.widgets.getAll()] else dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.widgets.get("container") if "container" in [w.name for w in dbutils.widgets.getAll()] else dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
RAW_PATH = f"{BASE_PATH}/raw"
BRONZE_PATH = f"{BASE_PATH}/bronze"

print(f"RAW_PATH: {RAW_PATH}")
print(f"BRONZE_PATH: {BRONZE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import Email Loaders

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/your-repo/mater_thesis/src")

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, ArrayType, IntegerType, BooleanType
from pyspark.sql.functions import col, current_timestamp, lit
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Email Schema for Delta Lake

# COMMAND ----------

# Email schema for Bronze zone
EMAIL_SCHEMA = StructType([
    StructField("message_id", StringType(), False),
    StructField("subject", StringType(), True),
    StructField("sender_name", StringType(), True),
    StructField("sender_email", StringType(), True),
    StructField("recipients", ArrayType(StringType()), True),
    StructField("cc", ArrayType(StringType()), True),
    StructField("bcc", ArrayType(StringType()), True),
    StructField("date", TimestampType(), True),
    StructField("body_text", StringType(), True),
    StructField("body_html", StringType(), True),
    StructField("conversation_id", StringType(), True),
    StructField("in_reply_to", StringType(), True),
    StructField("references", ArrayType(StringType()), True),
    StructField("has_attachments", BooleanType(), True),
    StructField("attachment_count", IntegerType(), True),
    StructField("attachment_names", ArrayType(StringType()), True),
    StructField("source_file", StringType(), True),
    StructField("source_path", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
])

# Attachment schema
ATTACHMENT_SCHEMA = StructType([
    StructField("attachment_id", StringType(), False),
    StructField("parent_message_id", StringType(), False),
    StructField("filename", StringType(), True),
    StructField("file_extension", StringType(), True),
    StructField("size_bytes", IntegerType(), True),
    StructField("content_type", StringType(), True),
    StructField("storage_path", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MSG File Loader (Using extract-msg)

# COMMAND ----------

import extract_msg

def load_msg_file(file_path: str, source_name: str = None) -> Dict[str, Any]:
    """
    Load a single MSG file and return structured data.

    Args:
        file_path: Path to MSG file
        source_name: Original source file name

    Returns:
        Dictionary with email data matching EMAIL_SCHEMA
    """
    msg = extract_msg.Message(file_path)

    try:
        # Parse date
        msg_date = None
        if msg.date:
            try:
                msg_date = msg.date
            except Exception:
                pass

        # Generate message ID
        message_id = msg.messageId or hashlib.md5(
            f"{msg.subject}{msg.date}{msg.sender}".encode()
        ).hexdigest()

        # Extract attachments info
        attachments = []
        for att in msg.attachments:
            attachments.append(att.longFilename or att.shortFilename or "unnamed")

        # Parse recipients
        recipients = [r.strip() for r in (msg.to or "").split(";") if r.strip()]
        cc = [r.strip() for r in (msg.cc or "").split(";") if r.strip()]
        bcc = [r.strip() for r in (msg.bcc or "").split(";") if r.strip()]

        return {
            "message_id": message_id,
            "subject": msg.subject or "(No Subject)",
            "sender_name": msg.sender or "Unknown",
            "sender_email": msg.senderEmail or "",
            "recipients": recipients,
            "cc": cc,
            "bcc": bcc,
            "date": msg_date,
            "body_text": msg.body or "",
            "body_html": msg.htmlBody,
            "conversation_id": None,  # Will be computed during threading
            "in_reply_to": None,
            "references": [],
            "has_attachments": len(attachments) > 0,
            "attachment_count": len(attachments),
            "attachment_names": attachments,
            "source_file": source_name or file_path.split("/")[-1],
            "source_path": file_path,
            "ingestion_timestamp": datetime.utcnow(),
        }

    finally:
        msg.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Discover Files in RAW Zone

# COMMAND ----------

def list_files_by_extension(base_path: str, extensions: List[str]) -> List[str]:
    """List all files with given extensions in path."""
    all_files = []

    try:
        items = dbutils.fs.ls(base_path)
        for item in items:
            if item.isDir():
                # Recurse into subdirectories
                all_files.extend(list_files_by_extension(item.path, extensions))
            else:
                # Check extension
                ext = item.path.lower().split(".")[-1] if "." in item.path else ""
                if ext in extensions:
                    all_files.append(item.path)
    except Exception as e:
        print(f"Error listing {base_path}: {e}")

    return all_files

# Discover email files
msg_files = list_files_by_extension(RAW_PATH, ["msg"])
pst_files = list_files_by_extension(RAW_PATH, ["pst"])

print(f"Found {len(msg_files)} MSG files")
print(f"Found {len(pst_files)} PST files")

# Sample files
if msg_files:
    print("\nSample MSG files:")
    for f in msg_files[:5]:
        print(f"  - {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Process MSG Files

# COMMAND ----------

def process_msg_files_batch(file_paths: List[str], batch_size: int = 100) -> List[Dict]:
    """
    Process MSG files in batches.

    Args:
        file_paths: List of MSG file paths
        batch_size: Number of files per batch

    Returns:
        List of email dictionaries
    """
    all_emails = []
    errors = []

    for i, file_path in enumerate(file_paths):
        try:
            # Download from ADLS to local temp
            local_path = f"/tmp/msg_{i}.msg"
            dbutils.fs.cp(file_path, f"file:{local_path}")

            # Load MSG file
            email_data = load_msg_file(local_path, file_path.split("/")[-1])
            email_data["source_path"] = file_path  # Keep original ADLS path
            all_emails.append(email_data)

            # Cleanup
            import os
            if os.path.exists(local_path):
                os.remove(local_path)

        except Exception as e:
            errors.append({"file": file_path, "error": str(e)})

        # Progress update
        if (i + 1) % batch_size == 0:
            print(f"Processed {i + 1}/{len(file_paths)} files...")

    print(f"\nProcessed {len(all_emails)} emails successfully")
    print(f"Errors: {len(errors)}")

    return all_emails, errors

# Process MSG files
if msg_files:
    emails, errors = process_msg_files_batch(msg_files)
else:
    emails, errors = [], []
    print("No MSG files found. Upload files to RAW zone first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Reconstruct Email Threads

# COMMAND ----------

def build_email_threads(emails: List[Dict]) -> List[Dict]:
    """
    Reconstruct conversation threads using email headers.

    Uses:
    - In-Reply-To header
    - References header
    - Subject matching (Re:, Fwd:)
    """
    # Index by message_id
    by_id = {e["message_id"]: e for e in emails}

    # Subject normalization for fallback matching
    def normalize_subject(subj: str) -> str:
        import re
        # Remove Re:, Fwd:, etc.
        return re.sub(r'^(re:|fwd?:|fw:)\s*', '', subj.lower(), flags=re.IGNORECASE).strip()

    # Group by normalized subject for fallback
    by_subject = {}
    for e in emails:
        norm_subj = normalize_subject(e["subject"])
        if norm_subj not in by_subject:
            by_subject[norm_subj] = []
        by_subject[norm_subj].append(e)

    # Assign conversation IDs
    for email in emails:
        if email["conversation_id"]:
            continue

        # Try In-Reply-To
        if email.get("in_reply_to") and email["in_reply_to"] in by_id:
            parent = by_id[email["in_reply_to"]]
            email["conversation_id"] = parent.get("conversation_id") or parent["message_id"]
            continue

        # Try References
        for ref in email.get("references", []):
            if ref in by_id:
                parent = by_id[ref]
                email["conversation_id"] = parent.get("conversation_id") or parent["message_id"]
                break

        # Fallback: match by subject
        if not email["conversation_id"]:
            norm_subj = normalize_subject(email["subject"])
            related = by_subject.get(norm_subj, [])
            if len(related) > 1:
                # Find earliest as root
                earliest = min(related, key=lambda x: x["date"] or datetime.max)
                email["conversation_id"] = earliest["message_id"]
            else:
                email["conversation_id"] = email["message_id"]

    return emails

# Build threads
if emails:
    emails = build_email_threads(emails)
    print(f"Threads reconstructed. Unique conversations: {len(set(e['conversation_id'] for e in emails))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write to Bronze Zone (Delta Lake)

# COMMAND ----------

def write_emails_to_bronze(emails: List[Dict], bronze_path: str):
    """Write emails to Bronze Delta Lake table."""
    if not emails:
        print("No emails to write")
        return

    # Convert to Spark DataFrame
    df = spark.createDataFrame(emails, schema=EMAIL_SCHEMA)

    # Add metadata
    df = df.withColumn("_ingested_at", current_timestamp())

    # Write to Delta
    table_path = f"{bronze_path}/emails"

    # Check if table exists
    try:
        delta_table = DeltaTable.forPath(spark, table_path)
        # Merge (upsert) based on message_id
        delta_table.alias("target").merge(
            df.alias("source"),
            "target.message_id = source.message_id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        print(f"Merged {df.count()} emails into {table_path}")
    except Exception:
        # Table doesn't exist, create it
        df.write.format("delta").mode("overwrite").save(table_path)
        print(f"Created new Delta table with {df.count()} emails at {table_path}")

# Write to Bronze
if emails:
    write_emails_to_bronze(emails, BRONZE_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify Bronze Data

# COMMAND ----------

# Read back and verify
emails_df = spark.read.format("delta").load(f"{BRONZE_PATH}/emails")

print(f"Total emails in Bronze: {emails_df.count()}")
print(f"Unique conversations: {emails_df.select('conversation_id').distinct().count()}")
print(f"Emails with attachments: {emails_df.filter(col('has_attachments') == True).count()}")

# Show sample
display(emails_df.select("message_id", "subject", "sender_name", "date", "conversation_id", "has_attachments").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Email Statistics

# COMMAND ----------

from pyspark.sql.functions import year, month, count, avg

# Emails by year
print("Emails by Year:")
emails_df.groupBy(year("date").alias("year")).count().orderBy("year").display()

# COMMAND ----------

# Emails by sender (top 10)
print("Top 10 Senders:")
emails_df.groupBy("sender_email").count().orderBy(col("count").desc()).limit(10).display()

# COMMAND ----------

# Thread size distribution
from pyspark.sql.functions import size

print("Thread Size Distribution:")
thread_sizes = emails_df.groupBy("conversation_id").count()
thread_sizes.groupBy("count").count().orderBy("count").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary & Next Steps

# COMMAND ----------

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║              EMAIL INGESTION COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Emails Processed  : {emails_df.count():<35} ║
║  Unique Conversations    : {emails_df.select('conversation_id').distinct().count():<35} ║
║  Emails with Attachments : {emails_df.filter(col('has_attachments')==True).count():<35} ║
║  Bronze Table Location   : {BRONZE_PATH}/emails                 ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS:                                                     ║
║  1. Run 02_document_ingestion.py for PDF, DOCX, XLSX files       ║
║  2. Run language detection pipeline                              ║
║  3. Proceed to chunking and NLP processing                       ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PST Processing (Alternative)
# MAGIC
# MAGIC For large PST files, use the cluster's bash shell:
# MAGIC
# MAGIC ```bash
# MAGIC %sh
# MAGIC # Install libpst
# MAGIC apt-get update && apt-get install -y pst-utils
# MAGIC
# MAGIC # Extract PST to EML files
# MAGIC readpst -e -o /tmp/pst_extract /dbfs/raw/archive.pst
# MAGIC ```
# MAGIC
# MAGIC Then process the extracted EML files similar to MSG files.
