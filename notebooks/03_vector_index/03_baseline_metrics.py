# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Baseline Metrics Evaluation
# MAGIC
# MAGIC **Phase 3: Vector Index & Basic RAG | Week 5**
# MAGIC
# MAGIC This notebook measures baseline retrieval and RAG performance metrics.
# MAGIC
# MAGIC ## Metrics Measured
# MAGIC - **MRR@10** (Mean Reciprocal Rank) - Target: > 0.55
# MAGIC - **NDCG@5/10** (Normalized Discounted Cumulative Gain)
# MAGIC - **Precision@K** and **Recall@K**
# MAGIC - **Hit Rate@K**
# MAGIC - **Latency** (P50, P95, P99)
# MAGIC
# MAGIC ## Success Criteria
# MAGIC **Milestone M3: Basic RAG operational; baseline MRR recorded**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai azure-search-documents numpy pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Azure credentials
OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")
SEARCH_ENDPOINT = dbutils.secrets.get("azure-search", "endpoint")
SEARCH_KEY = dbutils.secrets.get("azure-search", "api-key")
STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_PATH = f"{BASE_PATH}/gold"

INDEX_NAME = "knowledge-chunks"

print(f"Evaluating index: {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Clients

# COMMAND ----------

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import numpy as np
import time
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime

# Initialize search client
search_client = SearchClient(
    SEARCH_ENDPOINT,
    INDEX_NAME,
    AzureKeyCredential(SEARCH_KEY)
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
)

# Initialize LLM (for generating test queries)
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="gpt-4o",
    temperature=0.7,
)

print("Clients initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Evaluation Metrics

# COMMAND ----------

class RetrievalMetrics:
    """Calculate retrieval quality metrics."""

    @staticmethod
    def reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """Calculate reciprocal rank (1/position of first relevant)."""
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """Calculate DCG@K."""
        relevances = relevances[:k]
        if not relevances:
            return 0.0
        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / np.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """Calculate NDCG@K."""
        dcg = RetrievalMetrics.dcg_at_k(relevances, k)
        ideal = sorted(relevances, reverse=True)
        idcg = RetrievalMetrics.dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_k & relevant_set) / k

    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_k & relevant_set) / len(relevant_set)

    @staticmethod
    def hit_rate_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Hit Rate@K (1 if any relevant in top-k)."""
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return 1.0 if retrieved_k & relevant_set else 0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Test Queries
# MAGIC
# MAGIC We'll create test queries based on actual document content to have ground truth.

# COMMAND ----------

def get_sample_documents(n: int = 50) -> List[Dict]:
    """Get sample documents from index for test query generation."""
    results = search_client.search(
        search_text="*",
        top=n,
        select=["chunk_id", "content", "source_file", "parent_type"],
    )
    return list(results)

# Get sample documents
sample_docs = get_sample_documents(50)
print(f"Retrieved {len(sample_docs)} sample documents for test generation")

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate

def generate_test_query(document: Dict) -> Dict[str, Any]:
    """Generate a test query from a document with the document as ground truth."""
    content = document.get("content", "")[:2000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a natural question that would be answered by the given document content. Return only the question, nothing else."),
        ("human", "Document content:\n{content}\n\nGenerate a question:"),
    ])

    chain = prompt | llm
    response = chain.invoke({"content": content})
    question = response.content.strip()

    return {
        "query": question,
        "relevant_docs": [document["chunk_id"]],
        "source_file": document.get("source_file", ""),
    }

# Generate test queries
print("Generating test queries...")
test_queries = []

for i, doc in enumerate(sample_docs[:30]):  # Use 30 documents for testing
    try:
        test_case = generate_test_query(doc)
        test_queries.append(test_case)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} test queries...")
    except Exception as e:
        print(f"Error generating query for doc {i}: {e}")

print(f"\nGenerated {len(test_queries)} test queries")

# COMMAND ----------

# Display sample test queries
print("Sample Test Queries:")
for i, tq in enumerate(test_queries[:5], 1):
    print(f"\n[{i}] Query: {tq['query'][:100]}...")
    print(f"    Ground Truth: {tq['relevant_docs'][0]}")
    print(f"    Source: {tq['source_file']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run Retrieval Evaluation

# COMMAND ----------

def evaluate_search_type(test_queries: List[Dict], search_type: str, top_k: int = 10) -> Dict[str, Any]:
    """Evaluate a specific search type."""
    print(f"\nEvaluating {search_type} search...")

    mrr_scores = []
    ndcg_5_scores = []
    ndcg_10_scores = []
    precision_1_scores = []
    precision_5_scores = []
    precision_10_scores = []
    recall_5_scores = []
    recall_10_scores = []
    hit_1_scores = []
    hit_5_scores = []
    hit_10_scores = []
    retrieval_times = []

    for test_case in test_queries:
        query = test_case["query"]
        relevant_ids = test_case["relevant_docs"]

        # Execute search
        start_time = time.time()

        if search_type == "vector":
            query_embedding = embeddings.embed_query(query)
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            results = search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top_k,
                select=["chunk_id"],
            )

        elif search_type == "keyword":
            results = search_client.search(
                search_text=query,
                top=top_k,
                select=["chunk_id"],
            )

        else:  # hybrid
            query_embedding = embeddings.embed_query(query)
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            results = search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                select=["chunk_id"],
                query_type="semantic",
                semantic_configuration_name="semantic-config",
            )

        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)

        # Get retrieved IDs
        retrieved_ids = [r["chunk_id"] for r in results]

        # Calculate binary relevance for NDCG
        relevant_set = set(relevant_ids)
        relevances = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved_ids]

        # Calculate metrics
        mrr_scores.append(RetrievalMetrics.reciprocal_rank(relevant_ids, retrieved_ids))
        ndcg_5_scores.append(RetrievalMetrics.ndcg_at_k(relevances, 5))
        ndcg_10_scores.append(RetrievalMetrics.ndcg_at_k(relevances, 10))
        precision_1_scores.append(RetrievalMetrics.precision_at_k(relevant_ids, retrieved_ids, 1))
        precision_5_scores.append(RetrievalMetrics.precision_at_k(relevant_ids, retrieved_ids, 5))
        precision_10_scores.append(RetrievalMetrics.precision_at_k(relevant_ids, retrieved_ids, 10))
        recall_5_scores.append(RetrievalMetrics.recall_at_k(relevant_ids, retrieved_ids, 5))
        recall_10_scores.append(RetrievalMetrics.recall_at_k(relevant_ids, retrieved_ids, 10))
        hit_1_scores.append(RetrievalMetrics.hit_rate_at_k(relevant_ids, retrieved_ids, 1))
        hit_5_scores.append(RetrievalMetrics.hit_rate_at_k(relevant_ids, retrieved_ids, 5))
        hit_10_scores.append(RetrievalMetrics.hit_rate_at_k(relevant_ids, retrieved_ids, 10))

    return {
        "search_type": search_type,
        "num_queries": len(test_queries),
        "metrics": {
            "mrr": np.mean(mrr_scores),
            "mrr_at_10": np.mean(mrr_scores),
            "ndcg_at_5": np.mean(ndcg_5_scores),
            "ndcg_at_10": np.mean(ndcg_10_scores),
            "precision_at_1": np.mean(precision_1_scores),
            "precision_at_5": np.mean(precision_5_scores),
            "precision_at_10": np.mean(precision_10_scores),
            "recall_at_5": np.mean(recall_5_scores),
            "recall_at_10": np.mean(recall_10_scores),
            "hit_rate_at_1": np.mean(hit_1_scores),
            "hit_rate_at_5": np.mean(hit_5_scores),
            "hit_rate_at_10": np.mean(hit_10_scores),
        },
        "latency": {
            "avg_ms": np.mean(retrieval_times),
            "p50_ms": np.percentile(retrieval_times, 50),
            "p95_ms": np.percentile(retrieval_times, 95),
            "p99_ms": np.percentile(retrieval_times, 99),
        }
    }

# Evaluate all search types
results = {}
for search_type in ["vector", "keyword", "hybrid"]:
    results[search_type] = evaluate_search_type(test_queries, search_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Display Results

# COMMAND ----------

import pandas as pd

# Create comparison DataFrame
comparison_data = []
for search_type, result in results.items():
    row = {"Search Type": search_type}
    row.update(result["metrics"])
    row.update({f"latency_{k}": v for k, v in result["latency"].items()})
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

# COMMAND ----------

# Detailed metrics display
for search_type, result in results.items():
    print(f"\n{'='*60}")
    print(f" {search_type.upper()} SEARCH RESULTS")
    print(f"{'='*60}")

    m = result["metrics"]
    l = result["latency"]

    print(f"\n  Retrieval Metrics:")
    print(f"    MRR@10:       {m['mrr_at_10']:.4f}")
    print(f"    NDCG@5:       {m['ndcg_at_5']:.4f}")
    print(f"    NDCG@10:      {m['ndcg_at_10']:.4f}")
    print(f"    Precision@1:  {m['precision_at_1']:.4f}")
    print(f"    Precision@5:  {m['precision_at_5']:.4f}")
    print(f"    Recall@5:     {m['recall_at_5']:.4f}")
    print(f"    Recall@10:    {m['recall_at_10']:.4f}")
    print(f"    Hit Rate@1:   {m['hit_rate_at_1']:.4f}")
    print(f"    Hit Rate@5:   {m['hit_rate_at_5']:.4f}")

    print(f"\n  Latency:")
    print(f"    Average:      {l['avg_ms']:.1f} ms")
    print(f"    P50:          {l['p50_ms']:.1f} ms")
    print(f"    P95:          {l['p95_ms']:.1f} ms")
    print(f"    P99:          {l['p99_ms']:.1f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Baseline Results

# COMMAND ----------

import json

# Prepare baseline results
baseline_results = {
    "evaluation_date": datetime.utcnow().isoformat(),
    "index_name": INDEX_NAME,
    "num_test_queries": len(test_queries),
    "results_by_search_type": results,
    "best_configuration": {
        "search_type": max(results.keys(), key=lambda k: results[k]["metrics"]["mrr"]),
        "mrr": max(r["metrics"]["mrr"] for r in results.values()),
    },
    "target_metrics": {
        "mrr_target": 0.55,
        "achieved": max(r["metrics"]["mrr"] for r in results.values()) >= 0.55,
    }
}

# Save to Gold zone
baseline_path = f"{GOLD_PATH}/baseline_metrics/baseline_v1.json"
dbutils.fs.put(baseline_path, json.dumps(baseline_results, indent=2, default=str), overwrite=True)
print(f"Saved baseline results to {baseline_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Baseline Comparison Table

# COMMAND ----------

# Create baseline comparison table for thesis
print("\n" + "=" * 80)
print(" BASELINE METRICS COMPARISON TABLE (for thesis)")
print("=" * 80)

print("\n| Metric        | Vector  | Keyword | Hybrid  | Target  |")
print("|---------------|---------|---------|---------|---------|")

metrics_to_show = [
    ("MRR@10", "mrr_at_10", 0.55),
    ("NDCG@5", "ndcg_at_5", None),
    ("NDCG@10", "ndcg_at_10", None),
    ("Precision@1", "precision_at_1", None),
    ("Precision@5", "precision_at_5", None),
    ("Recall@5", "recall_at_5", None),
    ("Hit Rate@5", "hit_rate_at_5", None),
]

for label, key, target in metrics_to_show:
    vector_val = results["vector"]["metrics"][key]
    keyword_val = results["keyword"]["metrics"][key]
    hybrid_val = results["hybrid"]["metrics"][key]
    target_str = f"{target:.2f}" if target else "-"
    print(f"| {label:<13} | {vector_val:.4f}  | {keyword_val:.4f}  | {hybrid_val:.4f}  | {target_str:<7} |")

print("\n| Latency (ms)  | Vector  | Keyword | Hybrid  |")
print("|---------------|---------|---------|---------|")
print(f"| Average       | {results['vector']['latency']['avg_ms']:.1f}    | {results['keyword']['latency']['avg_ms']:.1f}    | {results['hybrid']['latency']['avg_ms']:.1f}    |")
print(f"| P95           | {results['vector']['latency']['p95_ms']:.1f}    | {results['keyword']['latency']['p95_ms']:.1f}    | {results['hybrid']['latency']['p95_ms']:.1f}    |")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Phase 3 Completion Summary

# COMMAND ----------

best_type = baseline_results["best_configuration"]["search_type"]
best_mrr = baseline_results["best_configuration"]["mrr"]

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║         PHASE 3: VECTOR INDEX & BASIC RAG COMPLETE               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─ BASELINE METRICS RECORDED ────────────────────────────────┐  ║
║  │                                                            │  ║
║  │  Best Configuration: {best_type.upper():<37} │  ║
║  │  Best MRR@10: {best_mrr:.4f}                                   │  ║
║  │  Target MRR: 0.55                                          │  ║
║  │  Status: {'✓ ACHIEVED' if best_mrr >= 0.55 else '○ Below Target':<47} │  ║
║  │                                                            │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  SEARCH TYPE COMPARISON:                                         ║
║  • Vector:  MRR={results['vector']['metrics']['mrr']:.4f}, Latency={results['vector']['latency']['avg_ms']:.0f}ms              ║
║  • Keyword: MRR={results['keyword']['metrics']['mrr']:.4f}, Latency={results['keyword']['latency']['avg_ms']:.0f}ms              ║
║  • Hybrid:  MRR={results['hybrid']['metrics']['mrr']:.4f}, Latency={results['hybrid']['latency']['avg_ms']:.0f}ms              ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ MILESTONE M3 COMPLETE                                         ║
║    • Basic RAG operational                                       ║
║    • Baseline MRR recorded                                       ║
║    • Ready for GraphRAG enhancement                              ║
╠══════════════════════════════════════════════════════════════════╣
║  COMPARATIVE EVALUATION FRAMEWORK:                               ║
║  Configuration      | Components           | Expected MRR       ║
║  ─────────────────────────────────────────────────────────────── ║
║  Baseline           | Vector search only   | ~0.55              ║
║  +GraphRAG          | + Graph + Communities| ~0.65              ║
║  +ReAct             | + ReAct Agent        | ~0.60              ║
║  Full System        | All components       | ~0.70              ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT PHASE: GraphRAG Construction (Weeks 6-8)                   ║
║  Run: notebooks/04_graphrag/01_entity_extraction.py              ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
