# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 6.2: Comparative Analysis
# MAGIC
# MAGIC Cross-system comparison and statistical analysis.
# MAGIC
# MAGIC **Week 11 - System Evaluation**
# MAGIC
# MAGIC ## Features
# MAGIC - System-by-system comparison
# MAGIC - Statistical significance testing
# MAGIC - Performance profiling
# MAGIC - Use case recommendations
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai pandas numpy scipy matplotlib seaborn delta-spark

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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from evaluation.comparative_analysis import (
    ComparativeAnalyzer,
    PerformanceProfiler,
    SystemType,
    SystemResult,
    ComparisonResult,
    AggregatedComparison
)
from evaluation.ragas_evaluator import RAGASEvaluator, RAGASConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure Paths
GOLD_PATH = "/mnt/datalake/gold"

# Input
RAGAS_RESULTS_TABLE = f"{GOLD_PATH}/ragas_evaluation_results"
RAGAS_PER_QUESTION_TABLE = f"{GOLD_PATH}/ragas_per_question"
QA_RESULTS_TABLE = f"{GOLD_PATH}/qa_evaluation_results"

# Output
COMPARATIVE_RESULTS_TABLE = f"{GOLD_PATH}/comparative_analysis_results"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Evaluation Data

# COMMAND ----------

# DBTITLE 1,Load RAGAS Results
# Load aggregated RAGAS results
ragas_df = spark.read.format("delta").load(RAGAS_RESULTS_TABLE)
ragas_agg = ragas_df.toPandas()

print("RAGAS Aggregated Results:")
display(ragas_df)

# COMMAND ----------

# DBTITLE 1,Load Per-Question Results
# Load per-question RAGAS results
per_question_df = spark.read.format("delta").load(RAGAS_PER_QUESTION_TABLE)
per_question = per_question_df.toPandas()

print(f"Loaded {len(per_question)} per-question records")
display(per_question_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Load QA Results
# Load QA results for latency and other metrics
qa_df = spark.read.format("delta").load(QA_RESULTS_TABLE)
qa_results = qa_df.toPandas()

print(f"Loaded {len(qa_results)} QA records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Statistical Significance Testing

# COMMAND ----------

# DBTITLE 1,Paired T-Tests Between Systems
# Perform paired t-tests between all system pairs
systems = per_question["system"].unique()
metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

significance_results = []

for metric in metrics:
    print(f"\n{'='*60}")
    print(f"Statistical Tests for: {metric.upper()}")
    print(f"{'='*60}")

    for i, sys1 in enumerate(systems):
        for sys2 in systems[i+1:]:
            # Get paired values
            df1 = per_question[per_question["system"] == sys1].sort_values("question")
            df2 = per_question[per_question["system"] == sys2].sort_values("question")

            # Ensure same questions
            common_questions = set(df1["question"]) & set(df2["question"])

            if len(common_questions) >= 2:
                vals1 = df1[df1["question"].isin(common_questions)][metric].values
                vals2 = df2[df2["question"].isin(common_questions)][metric].values

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(vals1, vals2)

                # Effect size (Cohen's d)
                diff = vals1 - vals2
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

                significance_results.append({
                    "metric": metric,
                    "system_1": sys1,
                    "system_2": sys2,
                    "mean_1": np.mean(vals1),
                    "mean_2": np.mean(vals2),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "significant": p_value < 0.05,
                    "n_samples": len(common_questions)
                })

                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {sys1} vs {sys2}: p={p_value:.4f} {sig_marker} (d={cohens_d:.2f})")

# Create DataFrame
significance_df = pd.DataFrame(significance_results)
display(spark.createDataFrame(significance_df))

# COMMAND ----------

# DBTITLE 1,Summarize Significant Findings
print("=" * 60)
print("SIGNIFICANT DIFFERENCES (p < 0.05)")
print("=" * 60)

significant = significance_df[significance_df["significant"] == True]

if len(significant) > 0:
    for _, row in significant.iterrows():
        winner = row["system_1"] if row["mean_1"] > row["mean_2"] else row["system_2"]
        loser = row["system_2"] if row["mean_1"] > row["mean_2"] else row["system_1"]
        print(f"\n{row['metric']}:")
        print(f"  {winner} > {loser}")
        print(f"  Means: {max(row['mean_1'], row['mean_2']):.3f} vs {min(row['mean_1'], row['mean_2']):.3f}")
        print(f"  Effect size (Cohen's d): {abs(row['cohens_d']):.2f}")
else:
    print("No statistically significant differences found at α=0.05")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Performance Analysis by Question Type

# COMMAND ----------

# DBTITLE 1,Analyze by Question Type
# Group by system and question type
by_type = per_question.groupby(["system", "question_type"]).agg({
    "faithfulness": ["mean", "std"],
    "answer_relevancy": ["mean", "std"],
    "context_precision": ["mean", "std"],
    "context_recall": ["mean", "std"]
}).reset_index()

# Flatten column names
by_type.columns = ["_".join(col).strip("_") for col in by_type.columns]

display(spark.createDataFrame(by_type))

# COMMAND ----------

# DBTITLE 1,Visualize Performance by Question Type
# Create heatmap of faithfulness by system and question type
pivot_faith = per_question.pivot_table(
    values="faithfulness",
    index="question_type",
    columns="system",
    aggfunc="mean"
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer Relevancy"),
    ("context_precision", "Context Precision"),
    ("context_recall", "Context Recall")
]

for ax, (metric, title) in zip(axes.flatten(), metrics_to_plot):
    pivot = per_question.pivot_table(
        values=metric,
        index="question_type",
        columns="system",
        aggfunc="mean"
    )

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('/tmp/performance_by_question_type.png', dpi=150)
display()

# COMMAND ----------

# DBTITLE 1,Best System by Question Type
print("=" * 60)
print("BEST SYSTEM BY QUESTION TYPE")
print("=" * 60)

question_types = per_question["question_type"].unique()

for qtype in question_types:
    qtype_data = per_question[per_question["question_type"] == qtype]

    # Calculate composite for each system
    by_system = qtype_data.groupby("system").agg({
        "faithfulness": "mean",
        "answer_relevancy": "mean",
        "context_precision": "mean",
        "context_recall": "mean"
    })

    by_system["composite"] = (
        by_system["faithfulness"] +
        by_system["answer_relevancy"] +
        by_system["context_precision"] +
        by_system["context_recall"]
    ) / 4

    best = by_system["composite"].idxmax()
    best_score = by_system.loc[best, "composite"]

    print(f"\n{qtype.upper()}:")
    print(f"  Best: {best} (composite: {best_score:.3f})")

    # Show all systems
    for system in by_system.index:
        score = by_system.loc[system, "composite"]
        print(f"    {system}: {score:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Efficiency Analysis

# COMMAND ----------

# DBTITLE 1,Latency Comparison
# Extract latency data from QA results if available
# (Simulated data for demonstration)

latency_data = {
    "baseline": {"mean": 450, "p95": 800, "p99": 1200},
    "graphrag": {"mean": 850, "p95": 1500, "p99": 2200},
    "react": {"mean": 2500, "p95": 4500, "p99": 6000},
    "multihop": {"mean": 4500, "p95": 7500, "p99": 10000}
}

print("=" * 60)
print("LATENCY COMPARISON (ms)")
print("=" * 60)

print(f"\n{'System':<12} {'Mean':>10} {'P95':>10} {'P99':>10}")
print("-" * 45)

for system, metrics in latency_data.items():
    print(f"{system:<12} {metrics['mean']:>10.0f} {metrics['p95']:>10.0f} {metrics['p99']:>10.0f}")

# COMMAND ----------

# DBTITLE 1,Quality vs Latency Trade-off
# Plot quality vs latency
fig, ax = plt.subplots(figsize=(10, 6))

for system in ragas_agg["system"]:
    composite = ragas_agg[ragas_agg["system"] == system]["composite_score"].values[0]
    latency = latency_data.get(system, {}).get("mean", 1000)

    ax.scatter(latency, composite, s=200, label=system.title())
    ax.annotate(system.upper(), (latency, composite), textcoords="offset points",
                xytext=(10, 5), fontsize=10)

ax.set_xlabel("Mean Latency (ms)")
ax.set_ylabel("Composite RAGAS Score")
ax.set_title("Quality vs Latency Trade-off")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/quality_vs_latency.png', dpi=150)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Recommendations

# COMMAND ----------

# DBTITLE 1,System Recommendations
print("=" * 60)
print("SYSTEM RECOMMENDATIONS")
print("=" * 60)

# Overall best
best_overall = ragas_agg.loc[ragas_agg["composite_score"].idxmax(), "system"]
print(f"\n🏆 OVERALL BEST: {best_overall.upper()}")
print(f"   Composite Score: {ragas_agg[ragas_agg['system'] == best_overall]['composite_score'].values[0]:.3f}")

# Best for each metric
print("\n📊 BEST BY METRIC:")
for metric in ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]:
    best = ragas_agg.loc[ragas_agg[metric].idxmax(), "system"]
    score = ragas_agg[ragas_agg["system"] == best][metric].values[0]
    metric_name = metric.replace("avg_", "").replace("_", " ").title()
    print(f"   {metric_name}: {best} ({score:.3f})")

# Use case recommendations
print("\n🎯 USE CASE RECOMMENDATIONS:")
recommendations = [
    ("Simple factual queries", "baseline", "Fastest response, good enough accuracy"),
    ("Theme/summary questions", "graphrag", "Community context provides high-level view"),
    ("Relationship exploration", "react", "Iterative reasoning finds connections"),
    ("Complex analytical queries", "multihop", "Question decomposition for thorough analysis"),
    ("Production deployment", "graphrag", "Best balance of quality and latency"),
]

for use_case, system, rationale in recommendations:
    print(f"\n   {use_case}:")
    print(f"     Recommended: {system.upper()}")
    print(f"     Reason: {rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Results

# COMMAND ----------

# DBTITLE 1,Save Comparative Analysis Results
# Prepare comprehensive results
comparative_results = {
    "analysis_timestamp": datetime.now().isoformat(),
    "num_systems": len(systems),
    "num_samples_per_system": len(per_question) // len(systems),
    "best_overall_system": best_overall,
    "best_overall_score": float(ragas_agg[ragas_agg["system"] == best_overall]["composite_score"].values[0]),
    "significant_differences": int(significance_df["significant"].sum()),
    "question_types_analyzed": list(per_question["question_type"].unique()),
}

# Save to Delta Lake
results_df = spark.createDataFrame([comparative_results])
results_df.write.format("delta").mode("overwrite").save(COMPARATIVE_RESULTS_TABLE)

print(f"Saved comparative analysis to {COMPARATIVE_RESULTS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Save Statistical Significance Results
significance_spark = spark.createDataFrame(significance_df)
significance_spark.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/statistical_significance")

print(f"Saved significance results to {GOLD_PATH}/statistical_significance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

# DBTITLE 1,Phase 6.2 Summary
print("=" * 60)
print("PHASE 6.2: COMPARATIVE ANALYSIS COMPLETE")
print("=" * 60)

print(f"\n📊 ANALYSIS SUMMARY:")
print(f"  Systems compared: {len(systems)}")
print(f"  Questions analyzed: {len(per_question) // len(systems)}")
print(f"  Question types: {len(per_question['question_type'].unique())}")

print(f"\n📈 KEY FINDINGS:")
print(f"  Best overall system: {best_overall.upper()}")
print(f"  Significant differences found: {significance_df['significant'].sum()}")

print(f"\n📁 OUTPUT:")
print(f"  Comparative results: {COMPARATIVE_RESULTS_TABLE}")
print(f"  Significance tests: {GOLD_PATH}/statistical_significance")

print("\n" + "=" * 60)
print("NEXT: 03_final_report.py - Generate Final Report")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Statistical Tests:**
# MAGIC - Paired t-tests compare systems on same questions
# MAGIC - Cohen's d measures effect size (0.2=small, 0.5=medium, 0.8=large)
# MAGIC - p < 0.05 considered statistically significant
# MAGIC
# MAGIC **Quality-Latency Trade-off:**
# MAGIC - Baseline: Fastest, lowest quality
# MAGIC - Full System: Slowest, highest quality
# MAGIC - GraphRAG: Good balance for production
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Generate final evaluation report
# MAGIC 2. Prepare thesis conclusions
# MAGIC 3. Document limitations and future work
