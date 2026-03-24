# Synapse Notebook: 02_thread_processing
# Processes Bronze emails → Silver layer (anonymization, KG extraction, summaries).
#
# Trigger: After 01_ingestion completes in Synapse Pipeline
# Input:   bronze/emails/, bronze/attachments/
# Output:  silver_llm/not_personal/email_chunks/, thread_summaries/, attachment_chunks/

# %% [markdown]
# # Phase 2: Bronze → Silver (Thread-Aware Processing)
# PII anonymization, KG extraction, summarization using Azure GPT-4o.

# %%
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from src.storage import ADLSAdapter

adapter = ADLSAdapter()

# %%
# --- Configuration ---
MODE = os.getenv("PIPELINE_MODE", "llm")  # llm | local | hybrid
BRONZE_PATH = "bronze"
SILVER_PATH = f"silver_{MODE}"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
KG_STRATEGY = "spacy"  # spacy | llm | hybrid
REL_STRATEGY = "cooccurrence"  # cooccurrence | llm | hybrid
PROCESS_ATTACHMENTS = os.getenv("PROCESS_ATTACHMENTS", "true").lower() in ("true", "1", "yes")

print(f"Mode: {MODE}")
print(f"Bronze: {BRONZE_PATH}")
print(f"Silver: {SILVER_PATH}")
print(f"Attachments: {PROCESS_ATTACHMENTS}")

# %%
# --- Download Bronze data to local temp for processing ---
# ThreadAwareProcessor expects local file paths, so we sync Bronze to temp.
import shutil

LOCAL_BRONZE = "/tmp/pipeline/bronze"
LOCAL_SILVER = "/tmp/pipeline/silver"

# Clean previous run
for d in [LOCAL_BRONZE, LOCAL_SILVER]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Download Bronze emails
email_files = adapter.list_files(f"{BRONZE_PATH}/emails", "*.json")
os.makedirs(f"{LOCAL_BRONZE}/emails", exist_ok=True)

for f in email_files:
    data = adapter.read_bytes(f)
    local_file = os.path.join(LOCAL_BRONZE, "emails", os.path.basename(f))
    with open(local_file, "wb") as fh:
        fh.write(data)

print(f"Downloaded {len(email_files)} Bronze emails to local temp")

# Download Bronze attachments if processing
if PROCESS_ATTACHMENTS:
    att_files = adapter.list_files(f"{BRONZE_PATH}/attachments", "*.json")
    os.makedirs(f"{LOCAL_BRONZE}/attachments", exist_ok=True)
    for f in att_files:
        data = adapter.read_bytes(f)
        local_file = os.path.join(LOCAL_BRONZE, "attachments", os.path.basename(f))
        with open(local_file, "wb") as fh:
            fh.write(data)
    print(f"Downloaded {len(att_files)} Bronze attachments")

# Download identity registry if exists
if adapter.exists("config/identity_registry.json"):
    reg_data = adapter.read_bytes("config/identity_registry.json")
    with open(f"{LOCAL_BRONZE}/identity_registry.json", "wb") as fh:
        fh.write(reg_data)
    print("Downloaded identity registry")

# %%
# --- Build identity registry ---
from src.silver.identity_registry import IdentityRegistry

registry_path = f"{LOCAL_BRONZE}/identity_registry.json"
registry = IdentityRegistry(registry_path)

# Build from Bronze emails
email_dir = Path(LOCAL_BRONZE) / "emails"
for email_file in email_dir.glob("*.json"):
    with open(email_file) as fh:
        email = json.load(fh)
    sender_email = email.get("sender_email", "")
    sender_name = email.get("sender_name", "")
    if sender_email:
        registry.register(sender_email, sender_name)
    for recip in email.get("recipients", []):
        if isinstance(recip, dict) and recip.get("email"):
            registry.register(recip["email"], recip.get("name", ""))

registry.save()
print(f"Identity registry: {len(registry)} identities")

# %%
# --- Run Thread-Aware Processing ---
from src.silver.thread_aware_processor import ThreadAwareProcessor

processor = ThreadAwareProcessor(
    bronze_path=LOCAL_BRONZE,
    silver_path=LOCAL_SILVER,
    mode=MODE,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    kg_strategy=KG_STRATEGY,
    rel_strategy=REL_STRATEGY,
    registry_path=registry_path,
)

stats = processor.process_all()
print(f"\nProcessing stats: {json.dumps(stats, indent=2, default=str)}")

# %%
# --- Upload Silver results to ADLS ---
logger.info("Uploading Silver layer to ADLS...")

upload_count = 0
silver_local = Path(LOCAL_SILVER)

for local_file in silver_local.rglob("*.json"):
    rel_path = local_file.relative_to(silver_local)
    adls_path = f"{SILVER_PATH}/{rel_path}"
    with open(local_file, "rb") as fh:
        adapter.write_bytes(adls_path, fh.read())
    upload_count += 1

# Upload updated identity registry
adapter.copy_local_to_adls = None  # Use write_bytes
with open(registry_path, "rb") as fh:
    adapter.write_bytes("config/identity_registry.json", fh.read())

print(f"\nUploaded {upload_count} Silver files + identity registry to ADLS")

# %%
# --- Save processing stats ---
stats["timestamp"] = datetime.now().isoformat()
stats["mode"] = MODE
adapter.write_json(f"{SILVER_PATH}/processing_stats.json", stats)

print(f"\n{'='*50}")
print(f"Silver processing complete ({MODE} mode)")
print(f"  Output: {SILVER_PATH}/")
print(f"  Files:  {upload_count}")
print(f"{'='*50}")

# Clean up temp
shutil.rmtree("/tmp/pipeline", ignore_errors=True)

dbutils.notebook.exit(json.dumps(stats))  # type: ignore
