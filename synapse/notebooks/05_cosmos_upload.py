# Synapse Notebook: 05_cosmos_upload
# Uploads Gold layer data to Cosmos DB for fast querying.
# Run after 03_gold_indexing completes.
#
# Input:  gold_llm/ on ADLS (graph, communities, paths)
#         silver_llm/ on ADLS (chunks, thread summaries)
# Output: Cosmos DB Gremlin (graph) + NoSQL (chunks, communities, summaries)

# %% [markdown]
# # Phase 5: Upload to Cosmos DB
# Loads Gold graph into Cosmos DB Gremlin and Silver chunks into Cosmos DB NoSQL.

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
from src.storage.cosmos_adapter import CosmosAdapter

adapter = ADLSAdapter()

# %%
# --- Configuration ---
MODE = os.getenv("PIPELINE_MODE", "llm")
GOLD_PATH = f"gold_{MODE}"
SILVER_PATH = f"silver_{MODE}"

# Initialize Cosmos DB
cosmos = CosmosAdapter.from_env()

print(f"Cosmos DB Gremlin: {cosmos.gremlin_endpoint}")
print(f"Cosmos DB NoSQL:   {cosmos.nosql_endpoint}")

# %%
# --- Step 1: Create NoSQL containers (idempotent) ---
print("Creating NoSQL containers...")
cosmos.create_nosql_containers()
print("Containers ready.")

# %%
# --- Step 2: Upload Knowledge Graph to Gremlin ---
print("\n=== Uploading Knowledge Graph to Gremlin ===")

# Load nodes from ADLS
nodes_data = adapter.read_json(f"{GOLD_PATH}/knowledge_graph/nodes.json")
print(f"Nodes to upload: {len(nodes_data)}")

# Convert to list format for bulk upsert
nodes_list = []
for node_id, node in nodes_data.items():
    nodes_list.append({
        "node_id": node_id,
        "name": node.get("name", ""),
        "node_type": node.get("node_type", "UNKNOWN"),
        "properties": {
            "mention_count": node.get("mention_count", 0),
            "source_chunks": json.dumps(node.get("source_chunks", [])[:10]),
        },
    })

node_count = cosmos.bulk_upsert_nodes(nodes_list)
print(f"Uploaded {node_count} nodes to Gremlin")

# %%
# --- Step 3: Upload Edges to Gremlin ---
edges_data = adapter.read_json(f"{GOLD_PATH}/knowledge_graph/edges.json")
print(f"\nEdges to upload: {len(edges_data)}")

edges_list = []
for edge in edges_data:
    edges_list.append({
        "source_id": edge.get("source_id", ""),
        "target_id": edge.get("target_id", ""),
        "edge_type": edge.get("edge_type", "RELATED_TO"),
        "properties": {
            "weight": edge.get("weight", 1.0),
        },
    })

edge_count = cosmos.bulk_upsert_edges(edges_list)
print(f"Uploaded {edge_count} edges to Gremlin")

# %%
# --- Step 4: Upload PathRAG Paths to Gremlin ---
paths_file = f"{GOLD_PATH}/paths/path_index.json"
if adapter.exists(paths_file):
    paths_data = adapter.read_json(paths_file)
    print(f"\nPaths to upload: {len(paths_data)}")

    path_count = 0
    for i, path in enumerate(paths_data):
        try:
            cosmos.upsert_path(
                path_id=path.get("path_id", f"path_{i}"),
                source_id=path.get("source_id", ""),
                target_id=path.get("target_id", ""),
                path_nodes=path.get("node_ids", []),
                path_edges=path.get("edge_types", []),
                description=path.get("description", ""),
                weight=path.get("weight", 1.0),
            )
            path_count += 1
        except Exception as e:
            logger.warning(f"Failed to upload path {i}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"  Uploaded {i + 1}/{len(paths_data)} paths")

    print(f"Uploaded {path_count} paths to Gremlin")
else:
    print("No path index found, skipping.")

# %%
# --- Step 5: Upload Silver Chunks to NoSQL ---
print("\n=== Uploading Chunks to Cosmos DB NoSQL ===")

chunk_count = 0

# Email chunks
email_chunks = adapter.list_files(f"{SILVER_PATH}/not_personal/email_chunks", "*.json")
print(f"Email chunks: {len(email_chunks)}")

for f in email_chunks:
    try:
        chunk = adapter.read_json(f)
        chunk["id"] = chunk.get("chunk_id", Path(f).stem)
        chunk["partitionKey"] = chunk.get("thread_id", "unknown")
        cosmos.upsert_chunk(chunk)
        chunk_count += 1
    except Exception as e:
        logger.warning(f"Failed to upload chunk {f}: {e}")

    if chunk_count % 200 == 0 and chunk_count > 0:
        logger.info(f"  Uploaded {chunk_count} chunks...")

# Attachment chunks
att_chunks = adapter.list_files(f"{SILVER_PATH}/not_personal/attachment_chunks", "*.json")
print(f"Attachment chunks: {len(att_chunks)}")

for f in att_chunks:
    try:
        chunk = adapter.read_json(f)
        chunk["id"] = chunk.get("chunk_id", Path(f).stem)
        chunk["partitionKey"] = chunk.get("thread_id", "unknown")
        cosmos.upsert_chunk(chunk)
        chunk_count += 1
    except Exception as e:
        logger.warning(f"Failed to upload attachment chunk {f}: {e}")

print(f"Uploaded {chunk_count} total chunks to NoSQL")

# %%
# --- Step 6: Upload Communities to NoSQL ---
print("\n=== Uploading Communities ===")

comm_count = 0
for level_dir in ["level_0", "level_1", "level_2"]:
    comm_files = adapter.list_files(f"{GOLD_PATH}/communities/{level_dir}", "*.json")
    for f in comm_files:
        try:
            comm = adapter.read_json(f)
            comm["id"] = comm.get("community_id", Path(f).stem)
            comm["partitionKey"] = str(comm.get("level", 0))
            cosmos.upsert_community(comm)
            comm_count += 1
        except Exception as e:
            logger.warning(f"Failed to upload community {f}: {e}")

print(f"Uploaded {comm_count} communities to NoSQL")

# %%
# --- Step 7: Upload Thread Summaries to NoSQL ---
print("\n=== Uploading Thread Summaries ===")

summary_count = 0
summary_files = adapter.list_files(f"{SILVER_PATH}/not_personal/thread_summaries", "*.json")
print(f"Thread summaries: {len(summary_files)}")

for f in summary_files:
    try:
        summary = adapter.read_json(f)
        summary["id"] = summary.get("thread_id", Path(f).stem)
        summary["partitionKey"] = summary.get("thread_id", "unknown")
        cosmos.upsert_thread_summary(summary)
        summary_count += 1
    except Exception as e:
        logger.warning(f"Failed to upload summary {f}: {e}")

print(f"Uploaded {summary_count} thread summaries to NoSQL")

# %%
# --- Step 8: Verify ---
print("\n=== Verification ===")

stats = cosmos.get_graph_stats()
print(f"Gremlin graph:  {stats['node_count']} nodes, {stats['edge_count']} edges")
print(f"Node types:     {stats['node_types']}")

# Quick NoSQL check
test_chunks = cosmos.search_chunks_by_text("project", limit=3)
print(f"NoSQL test:     Found {len(test_chunks)} chunks matching 'project'")

# %%
# --- Save upload stats ---
upload_stats = {
    "timestamp": datetime.now().isoformat(),
    "mode": MODE,
    "gremlin_nodes": node_count,
    "gremlin_edges": edge_count,
    "gremlin_paths": path_count if adapter.exists(paths_file) else 0,
    "nosql_chunks": chunk_count,
    "nosql_communities": comm_count,
    "nosql_thread_summaries": summary_count,
    "graph_stats": stats,
}
adapter.write_json(f"{GOLD_PATH}/cosmos_upload_stats.json", upload_stats)

print(f"\n{'='*50}")
print(f"Cosmos DB upload complete")
print(f"  Graph:       {node_count} nodes, {edge_count} edges")
print(f"  Chunks:      {chunk_count}")
print(f"  Communities: {comm_count}")
print(f"  Summaries:   {summary_count}")
print(f"{'='*50}")

cosmos.close()

dbutils.notebook.exit(json.dumps(upload_stats))  # type: ignore
