# Synapse Notebook: 01_ingestion
# Picks up files from input/source/, processes to Bronze, moves to input/processed/.
#
# Trigger: Synapse Pipeline (manual or Event Grid on input/source/)
# Input:   input/source/*.pst, *.pdf, *.docx, *.xlsx, *.pptx, *.txt
# Output:  bronze/emails/, bronze/documents/, bronze/attachments/

# %% [markdown]
# # Phase 1: Ingestion → Bronze Layer
# Picks new files from `input/source/`, extracts content, saves to Bronze.

# %%
import os
import sys
import json
import logging
from pathlib import Path, PurePosixPath
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, ".")

from src.storage import ADLSAdapter

adapter = ADLSAdapter()

# %%
# --- Discover new files in input/source/ ---
PST_EXTENSIONS = {".pst"}
DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".html", ".rtf"}

source_files = adapter.list_files("input/source")
pst_files = [f for f in source_files if PurePosixPath(f).suffix.lower() in PST_EXTENSIONS]
doc_files = [f for f in source_files if PurePosixPath(f).suffix.lower() in DOC_EXTENSIONS]

print(f"Found {len(pst_files)} PST files, {len(doc_files)} documents in input/source/")

if not pst_files and not doc_files:
    print("No new files to process. Exiting.")
    # Exit cleanly for Synapse pipeline
    dbutils.notebook.exit("NO_NEW_FILES")  # type: ignore

# %%
# --- Move files to input/processing/ (prevents double-pickup) ---
processing_files = {"pst": [], "doc": []}

for f in pst_files:
    dest = f.replace("input/source/", "input/processing/")
    adapter.move(f, dest)
    processing_files["pst"].append(dest)

for f in doc_files:
    dest = f.replace("input/source/", "input/processing/")
    adapter.move(f, dest)
    processing_files["doc"].append(dest)

print(f"Moved {len(source_files)} files to input/processing/")

# %%
# --- Process PST files ---
from src.bronze.pst_extractor import PSTExtractor
from src.bronze.bronze_loader import BronzeLoader

bronze_path = "bronze"
stats = {"pst_emails": 0, "documents": 0, "attachments": 0, "errors": []}

for pst_path in processing_files["pst"]:
    filename = PurePosixPath(pst_path).name
    logger.info(f"Processing PST: {filename}")

    try:
        # Download PST to local temp (pypff needs local file access)
        local_pst = adapter.get_local_temp(pst_path)

        # Extract emails
        extractor = PSTExtractor()
        emails = extractor.extract(local_pst)
        logger.info(f"  Extracted {len(emails)} emails from {filename}")

        # Save each email to Bronze
        for email_data in emails:
            record_id = email_data.get("record_id", email_data.get("message_id", "unknown"))
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(record_id))[:100]
            adapter.write_json(f"{bronze_path}/emails/{safe_id}.json", email_data)
            stats["pst_emails"] += 1

        # Clean up temp file
        os.remove(local_pst)

    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")
        stats["errors"].append({"file": filename, "error": str(e)})

# %%
# --- Process documents ---
from src.bronze.document_parser import DocumentParser

parser = DocumentParser()

for doc_path in processing_files["doc"]:
    filename = PurePosixPath(doc_path).name
    logger.info(f"Processing document: {filename}")

    try:
        # Download to local temp
        local_doc = adapter.get_local_temp(doc_path)

        # Parse document
        doc_data = parser.parse(local_doc)
        if doc_data:
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[:100]
            adapter.write_json(f"{bronze_path}/documents/{safe_name}.json", doc_data)
            stats["documents"] += 1

        os.remove(local_doc)

    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")
        stats["errors"].append({"file": filename, "error": str(e)})

# %%
# --- Move processed files to input/processed/ ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for category in ["pst", "doc"]:
    for f in processing_files[category]:
        dest = f.replace("input/processing/", f"input/processed/{timestamp}/")
        adapter.move(f, dest)

print(f"\nMoved all files to input/processed/{timestamp}/")

# %%
# --- Save ingestion stats ---
stats["timestamp"] = datetime.now().isoformat()
stats["total_files"] = len(source_files)
adapter.write_json(f"{bronze_path}/ingestion_stats_{timestamp}.json", stats)

print(f"\n{'='*50}")
print(f"Ingestion complete:")
print(f"  PST emails: {stats['pst_emails']}")
print(f"  Documents:  {stats['documents']}")
print(f"  Errors:     {len(stats['errors'])}")
print(f"{'='*50}")

# Return status for Synapse pipeline
dbutils.notebook.exit(json.dumps(stats))  # type: ignore
