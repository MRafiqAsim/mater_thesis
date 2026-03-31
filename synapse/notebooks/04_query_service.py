# Synapse Notebook: 04_query_service
# Downloads Gold layer and launches the Gradio query UI.
# Can also run batch queries for evaluation.
#
# Trigger: Manual (interactive use) or after 03_gold_indexing for evaluation
# Input:   gold_llm/, silver_llm/
# Output:  Gradio UI on port 7861 or evaluation results

# %% [markdown]
# # Phase 4: Query & Retrieval Service
# Launch interactive Gradio UI or run batch evaluation queries.

# %%
import os
import sys
import json
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from src.storage import ADLSAdapter, CosmosAdapter

adapter = ADLSAdapter()
cosmos = CosmosAdapter.from_env() if os.getenv("COSMOS_GREMLIN_ENDPOINT") or os.getenv("COSMOS_NOSQL_ENDPOINT") else None

# %%
# --- Configuration ---
MODE = os.getenv("PIPELINE_MODE", "llm")
GOLD_PATH = f"gold_{MODE}"
SILVER_PATH = f"silver_{MODE}"
LOCAL_GOLD = f"/tmp/pipeline/gold_{MODE}"
LOCAL_SILVER = f"/tmp/pipeline/silver_{MODE}"

# %%
# --- Download Gold and Silver to local (retriever needs local paths) ---
def sync_adls_to_local(adls_prefix: str, local_dir: str):
    """Download all files from ADLS prefix to local directory."""
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    count = 0
    # Walk through ADLS directory recursively
    all_files = []
    try:
        for item in adapter.fs_client.get_paths(path=adls_prefix, recursive=True):
            if not item.is_directory:
                all_files.append(item.name)
    except Exception:
        # Fallback: try common subdirectories
        for subdir in ["knowledge_graph", "communities", "paths", "embeddings",
                       "not_personal/email_chunks", "not_personal/attachment_chunks",
                       "not_personal/thread_summaries"]:
            try:
                files = adapter.list_files(f"{adls_prefix}/{subdir}")
                all_files.extend(files)
            except Exception:
                pass

    for f in all_files:
        try:
            data = adapter.read_bytes(f)
            rel = f.replace(f"{adls_prefix}/", "")
            local_file = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, "wb") as fh:
                fh.write(data)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to download {f}: {e}")

    return count

if not adapter.is_local:
    gold_count = sync_adls_to_local(GOLD_PATH, LOCAL_GOLD)
    silver_count = sync_adls_to_local(SILVER_PATH, LOCAL_SILVER)
    print(f"Downloaded {gold_count} Gold files, {silver_count} Silver files")
else:
    LOCAL_GOLD = os.path.join(adapter._local_root, GOLD_PATH.replace(f"_{MODE}", f"_{MODE}"))
    LOCAL_SILVER = os.path.join(adapter._local_root, SILVER_PATH.replace(f"_{MODE}", f"_{MODE}"))
    print(f"Local mode: Gold={LOCAL_GOLD}, Silver={LOCAL_SILVER}")

# %%
# --- Option A: Launch Gradio UI (interactive) ---
# Uncomment to launch the web interface:

# from src.app import create_app, get_retriever
# import src.app as app_module
#
# app_module._gold_path = LOCAL_GOLD
# app_module._silver_path = LOCAL_SILVER
# app_module._mode = MODE
# app_module._cosmos_adapter = cosmos
#
# get_retriever()  # Pre-load
# gradio_app = create_app()
# gradio_app.launch(server_name="0.0.0.0", server_port=7861, share=True)

# %%
# --- Option B: Batch query for evaluation ---
from src.retrieval import HybridRetriever, RetrievalStrategy

retriever = HybridRetriever(LOCAL_GOLD, LOCAL_SILVER, mode=MODE, cosmos_adapter=cosmos)

# Example evaluation queries
eval_queries = [
    "What projects were discussed in the emails and who was involved?",
    "What technical issues were reported and how were they resolved?",
    "Who are the main contacts and what roles do they play?",
]

strategies = [
    RetrievalStrategy.VECTOR,
    RetrievalStrategy.PATHRAG,
    RetrievalStrategy.GRAPHRAG,
    RetrievalStrategy.HYBRID,
    RetrievalStrategy.REACT,
]

results = []
for query in eval_queries:
    for strategy in strategies:
        result = retriever.retrieve(query, strategy)
        results.append({
            "query": query,
            "strategy": strategy.value,
            "answer": result.answer,
            "confidence": result.confidence,
            "chunks": len(result.chunks),
            "time": result.execution_time,
            "is_grounded": result.is_grounded,
        })
        print(f"  [{strategy.value}] {query[:50]}... → {result.confidence:.0%} ({result.execution_time:.1f}s)")

# Save evaluation results
adapter.write_json(f"{GOLD_PATH}/eval_results.json", results)
print(f"\nSaved {len(results)} evaluation results")
