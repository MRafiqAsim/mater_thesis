# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 6.1: RAGAS Evaluation
# MAGIC
# MAGIC Evaluate RAG systems using the RAGAS framework.
# MAGIC
# MAGIC **Week 11 - System Evaluation**
# MAGIC
# MAGIC ## Metrics
# MAGIC - Faithfulness: Is the answer grounded in context?
# MAGIC - Answer Relevancy: Is the answer relevant to the question?
# MAGIC - Context Precision: How much context is relevant?
# MAGIC - Context Recall: Is all needed information retrieved?
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai ragas datasets pandas numpy scipy delta-spark

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGASDatasetBuilder,
    RAGASConfig,
    EvaluationSample,
    EvaluationResult,
    AggregatedResults
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="azure-openai", key="endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="azure-openai", key="api-key")

print("Azure credentials loaded")

# COMMAND ----------

# DBTITLE 1,Configure Paths
GOLD_PATH = "/mnt/datalake/gold"

# Input: QA evaluation results from Phase 5
QA_RESULTS_TABLE = f"{GOLD_PATH}/qa_evaluation_results"

# Output
RAGAS_RESULTS_TABLE = f"{GOLD_PATH}/ragas_evaluation_results"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize RAGAS Evaluator

# COMMAND ----------

# DBTITLE 1,Initialize LLM and Embeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# LLM for evaluation
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="gpt-4o",
    temperature=0.0,  # Deterministic for evaluation
)

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-01"
)

print("LLM and embeddings initialized")

# COMMAND ----------

# DBTITLE 1,Configure RAGAS Evaluator
ragas_config = RAGASConfig(
    model_deployment="gpt-4o",
    embedding_deployment="text-embedding-3-large",
    batch_size=10,
    max_retries=3,
    # Metric weights
    faithfulness_weight=0.25,
    answer_relevancy_weight=0.25,
    context_precision_weight=0.25,
    context_recall_weight=0.25
)

evaluator = RAGASEvaluator(
    llm=llm,
    embeddings=embeddings,
    config=ragas_config
)

print("RAGAS evaluator initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load QA Results

# COMMAND ----------

# DBTITLE 1,Load QA Results from Phase 5
# Load QA evaluation results
qa_df = spark.read.format("delta").load(QA_RESULTS_TABLE)

print(f"Loaded {qa_df.count()} QA evaluation records")
qa_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Prepare Evaluation Samples
# Convert to pandas for processing
qa_pandas = qa_df.toPandas()

# Prepare samples for each system
systems = ["baseline", "graphrag", "react", "multihop"]

evaluation_data = {}

for system in systems:
    samples = []

    for _, row in qa_pandas.iterrows():
        answer_col = f"{system}_answer"
        sources_col = f"{system}_sources"

        if answer_col in row and row[answer_col]:
            # Extract contexts (simplified - would come from actual retrieval)
            contexts = [row[answer_col][:500]]  # Placeholder

            samples.append(EvaluationSample(
                question=row["question"],
                answer=row[answer_col],
                contexts=contexts,
                ground_truth=None,  # Would need ground truth for full evaluation
                metadata={
                    "question_type": row.get("question_type", "unknown"),
                    "system": system
                }
            ))

    evaluation_data[system] = samples
    print(f"{system}: {len(samples)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run RAGAS Evaluation

# COMMAND ----------

# DBTITLE 1,Evaluate Each System
def progress_callback(current, total):
    if current % 5 == 0 or current == total:
        print(f"  Progress: {current}/{total}")

ragas_results = {}

for system, samples in evaluation_data.items():
    if not samples:
        print(f"Skipping {system} - no samples")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating: {system.upper()}")
    print(f"{'='*60}")

    results = evaluator.evaluate(
        samples=samples,
        include_correctness=False,  # No ground truth
        progress_callback=progress_callback
    )

    ragas_results[system] = results

    # Print summary
    print(f"\n{system.upper()} Results:")
    print(f"  Faithfulness:      {results.avg_faithfulness:.3f} ± {results.std_faithfulness:.3f}")
    print(f"  Answer Relevancy:  {results.avg_answer_relevancy:.3f} ± {results.std_answer_relevancy:.3f}")
    print(f"  Context Precision: {results.avg_context_precision:.3f} ± {results.std_context_precision:.3f}")
    print(f"  Context Recall:    {results.avg_context_recall:.3f} ± {results.std_context_recall:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Results

# COMMAND ----------

# DBTITLE 1,Create Comparison DataFrame
comparison_data = []

for system, results in ragas_results.items():
    comparison_data.append({
        "system": system,
        "num_samples": results.num_samples,
        "faithfulness": results.avg_faithfulness,
        "faithfulness_std": results.std_faithfulness,
        "answer_relevancy": results.avg_answer_relevancy,
        "answer_relevancy_std": results.std_answer_relevancy,
        "context_precision": results.avg_context_precision,
        "context_precision_std": results.std_context_precision,
        "context_recall": results.avg_context_recall,
        "context_recall_std": results.std_context_recall,
        "composite_score": evaluator.calculate_composite_score(results)
    })

comparison_df = pd.DataFrame(comparison_data)
display(spark.createDataFrame(comparison_df))

# COMMAND ----------

# DBTITLE 1,Visualize Metrics Comparison
import matplotlib.pyplot as plt

# Create bar chart comparing systems
metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, (system, results) in enumerate(ragas_results.items()):
    values = [
        results.avg_faithfulness,
        results.avg_answer_relevancy,
        results.avg_context_precision,
        results.avg_context_recall
    ]
    ax.bar(x + i*width, values, width, label=system.title())

ax.set_ylabel('Score')
ax.set_title('RAGAS Metrics by System')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall'])
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/tmp/ragas_comparison.png', dpi=150)
display()

# COMMAND ----------

# DBTITLE 1,Composite Score Ranking
# Rank systems by composite score
ranking = comparison_df.sort_values("composite_score", ascending=False)

print("=" * 60)
print("SYSTEM RANKING BY COMPOSITE SCORE")
print("=" * 60)

for i, (_, row) in enumerate(ranking.iterrows(), 1):
    print(f"\n{i}. {row['system'].upper()}")
    print(f"   Composite Score: {row['composite_score']:.3f}")
    print(f"   Faithfulness: {row['faithfulness']:.3f}")
    print(f"   Relevancy: {row['answer_relevancy']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detailed Analysis

# COMMAND ----------

# DBTITLE 1,Per-Question Analysis
# Analyze individual question performance
per_question_data = []

for system, results in ragas_results.items():
    for result in results.individual_results:
        per_question_data.append({
            "system": system,
            "question": result.question[:50] + "...",
            "question_type": result.metadata.get("question_type", "unknown"),
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall
        })

per_question_df = pd.DataFrame(per_question_data)

# Group by question type
by_question_type = per_question_df.groupby(["system", "question_type"]).agg({
    "faithfulness": "mean",
    "answer_relevancy": "mean",
    "context_precision": "mean",
    "context_recall": "mean"
}).reset_index()

display(spark.createDataFrame(by_question_type))

# COMMAND ----------

# DBTITLE 1,Identify Weak Points
# Find questions where systems performed poorly
low_threshold = 0.5

weak_points = []

for system, results in ragas_results.items():
    for result in results.individual_results:
        if result.faithfulness < low_threshold:
            weak_points.append({
                "system": system,
                "question": result.question,
                "metric": "faithfulness",
                "score": result.faithfulness
            })
        if result.answer_relevancy < low_threshold:
            weak_points.append({
                "system": system,
                "question": result.question,
                "metric": "answer_relevancy",
                "score": result.answer_relevancy
            })

if weak_points:
    print("Questions with Low Scores (< 0.5):")
    print("-" * 60)
    for wp in weak_points[:10]:
        print(f"  [{wp['system']}] {wp['metric']}: {wp['score']:.3f}")
        print(f"    Question: {wp['question'][:60]}...")
else:
    print("No questions with scores below 0.5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Results

# COMMAND ----------

# DBTITLE 1,Save RAGAS Results to Delta Lake
# Prepare results for saving
save_data = []

for system, results in ragas_results.items():
    save_data.append({
        "system": system,
        "num_samples": results.num_samples,
        "avg_faithfulness": float(results.avg_faithfulness),
        "avg_answer_relevancy": float(results.avg_answer_relevancy),
        "avg_context_precision": float(results.avg_context_precision),
        "avg_context_recall": float(results.avg_context_recall),
        "std_faithfulness": float(results.std_faithfulness),
        "std_answer_relevancy": float(results.std_answer_relevancy),
        "std_context_precision": float(results.std_context_precision),
        "std_context_recall": float(results.std_context_recall),
        "composite_score": float(evaluator.calculate_composite_score(results)),
        "evaluated_at": results.evaluated_at
    })

results_df = spark.createDataFrame(save_data)
results_df.write.format("delta").mode("overwrite").save(RAGAS_RESULTS_TABLE)

print(f"Saved RAGAS results to {RAGAS_RESULTS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Save Individual Results
# Save per-question results
per_question_spark = spark.createDataFrame(per_question_data)
per_question_spark.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/ragas_per_question")

print(f"Saved per-question results to {GOLD_PATH}/ragas_per_question")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

# DBTITLE 1,Phase 6.1 Summary
print("=" * 60)
print("PHASE 6.1: RAGAS EVALUATION COMPLETE")
print("=" * 60)

print(f"\n📊 EVALUATION SUMMARY:")
print(f"  Systems evaluated: {len(ragas_results)}")
print(f"  Total samples: {sum(r.num_samples for r in ragas_results.values())}")

print(f"\n🏆 RANKING BY COMPOSITE SCORE:")
for i, (_, row) in enumerate(ranking.iterrows(), 1):
    print(f"  {i}. {row['system'].upper()}: {row['composite_score']:.3f}")

print(f"\n📁 OUTPUT:")
print(f"  RAGAS results: {RAGAS_RESULTS_TABLE}")
print(f"  Per-question: {GOLD_PATH}/ragas_per_question")

print("\n" + "=" * 60)
print("NEXT: 02_comparative_analysis.py - Cross-System Comparison")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **RAGAS Metrics:**
# MAGIC - **Faithfulness**: Measures factual consistency of answer with retrieved context
# MAGIC - **Answer Relevancy**: Measures how well the answer addresses the question
# MAGIC - **Context Precision**: Measures relevance of retrieved context
# MAGIC - **Context Recall**: Measures coverage of needed information in context
# MAGIC
# MAGIC **Interpretation:**
# MAGIC - Scores range from 0 to 1 (higher is better)
# MAGIC - Composite score is weighted average of all metrics
# MAGIC - Standard deviation indicates consistency across questions
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Run comparative analysis across systems
# MAGIC 2. Generate final evaluation report
# MAGIC 3. Prepare thesis conclusions
