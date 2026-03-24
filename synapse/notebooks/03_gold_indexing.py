# Synapse Notebook: 03_gold_indexing
# Builds Knowledge Graph, communities, paths, and embeddings from Silver layer.
#
# Trigger: After 02_thread_processing completes in Synapse Pipeline
# Input:   silver_llm/not_personal/email_chunks/, attachment_chunks/
# Output:  gold_llm/knowledge_graph/, communities/, paths/, embeddings/

# %% [markdown]
# # Phase 3: Silver → Gold (Knowledge Graph & Indexing)
# Graph construction, Leiden communities, PathRAG paths, vector embeddings.

# %%
import os
import sys
import json
import shutil
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
MODE = os.getenv("PIPELINE_MODE", "llm")
SILVER_PATH = f"silver_{MODE}"
GOLD_PATH = f"gold_{MODE}"
COMMUNITY_LEVELS = 3
MAX_PATH_LENGTH = 5
EMBEDDING_MODEL = "text-embedding-3-small"

print(f"Mode: {MODE}")
print(f"Silver: {SILVER_PATH}")
print(f"Gold: {GOLD_PATH}")

# %%
# --- Download Silver to local temp ---
LOCAL_SILVER = "/tmp/pipeline/silver"
LOCAL_GOLD = "/tmp/pipeline/gold"

for d in [LOCAL_SILVER, LOCAL_GOLD]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Download Silver chunks (email + attachment)
download_count = 0
for pattern in ["not_personal/email_chunks/*.json", "not_personal/attachment_chunks/*.json",
                "not_personal/thread_summaries/*.json"]:
    # List files matching the directory portion
    dir_part = pattern.rsplit("/", 1)[0]
    files = adapter.list_files(f"{SILVER_PATH}/{dir_part}", "*.json")
    if files:
        local_dir = os.path.join(LOCAL_SILVER, dir_part)
        os.makedirs(local_dir, exist_ok=True)
        for f in files:
            data = adapter.read_bytes(f)
            local_file = os.path.join(LOCAL_SILVER, f.replace(f"{SILVER_PATH}/", ""))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, "wb") as fh:
                fh.write(data)
            download_count += 1

print(f"Downloaded {download_count} Silver files to local temp")

# %%
# --- Step 1: Build Knowledge Graph ---
from src.gold.graph_builder import GraphBuilder

logger.info("Building knowledge graph...")
builder = GraphBuilder(silver_path=LOCAL_SILVER, gold_path=LOCAL_GOLD)
graph_stats = builder.build()

print(f"Graph: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges")

# %%
# --- Step 2: Detect Communities ---
from src.gold.community_detector import CommunityDetector

logger.info("Detecting communities...")
detector = CommunityDetector(gold_path=LOCAL_GOLD, mode=MODE)
comm_stats = detector.detect(levels=COMMUNITY_LEVELS)

print(f"Communities: {comm_stats.get('total_communities', 0)} across {COMMUNITY_LEVELS} levels")

# %%
# --- Step 3: Build Path Index ---
from src.gold.path_indexer import PathIndexer

logger.info("Building path index...")
indexer = PathIndexer(gold_path=LOCAL_GOLD)
path_stats = indexer.index(max_length=MAX_PATH_LENGTH)

print(f"Paths: {path_stats.get('total_paths', 0)} paths")

# %%
# --- Step 4: Generate Embeddings ---
from src.gold.embedding_generator import EmbeddingGenerator, EmbeddingConfig

logger.info("Generating embeddings...")
emb_config = EmbeddingConfig(model=EMBEDDING_MODEL)
generator = EmbeddingGenerator(gold_path=LOCAL_GOLD, config=emb_config, mode=MODE)

chunk_ids, chunk_embs = generator.embed_chunks(silver_path=LOCAL_SILVER)
entity_ids, entity_embs = generator.embed_entities()

print(f"Embeddings: {len(chunk_ids)} chunks, {len(entity_ids)} entities")

# %%
# --- Upload Gold to ADLS ---
logger.info("Uploading Gold layer to ADLS...")

upload_count = 0
gold_local = Path(LOCAL_GOLD)

for local_file in gold_local.rglob("*"):
    if local_file.is_file():
        rel_path = local_file.relative_to(gold_local)
        adls_path = f"{GOLD_PATH}/{rel_path}"
        with open(local_file, "rb") as fh:
            adapter.write_bytes(adls_path, fh.read())
        upload_count += 1

print(f"Uploaded {upload_count} Gold files to ADLS")

# %%
# --- Save Gold stats ---
all_stats = {
    "timestamp": datetime.now().isoformat(),
    "mode": MODE,
    "graph": graph_stats,
    "communities": comm_stats,
    "paths": path_stats,
    "embeddings": {"chunks": len(chunk_ids), "entities": len(entity_ids)},
}
adapter.write_json(f"{GOLD_PATH}/gold_stats.json", all_stats)

print(f"\n{'='*50}")
print(f"Gold indexing complete ({MODE} mode)")
print(f"  Nodes:       {graph_stats.get('nodes', 0)}")
print(f"  Edges:       {graph_stats.get('edges', 0)}")
print(f"  Communities: {comm_stats.get('total_communities', 0)}")
print(f"  Paths:       {path_stats.get('total_paths', 0)}")
print(f"  Embeddings:  {len(chunk_ids)} + {len(entity_ids)}")
print(f"{'='*50}")

shutil.rmtree("/tmp/pipeline", ignore_errors=True)

dbutils.notebook.exit(json.dumps(all_stats))  # type: ignore
