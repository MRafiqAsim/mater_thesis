# Phase 6: Evaluation

**Duration:** Week 11
**Goal:** Measure quality and generate final report for thesis

---

## Overview

### What We're Doing

In this phase, we rigorously evaluate our system:
1. Run RAGAS evaluation framework
2. Compare all 4 systems statistically
3. Analyze performance by question type
4. Generate publication-ready thesis report

### Why This Matters

- **Thesis Requirement**: Need quantitative evidence that GraphRAG + ReAct improves retrieval
- **Scientific Rigor**: Statistical significance proves improvements aren't random
- **Publication Quality**: Professional reports and visualizations

### The 4 Systems We Compare

| System | Description | Expected Strength |
|--------|-------------|-------------------|
| **Baseline RAG** | Vector search + GPT-4o | Simple factual queries |
| **GraphRAG Only** | Graph + community search | Relationship queries |
| **ReAct Only** | Agent + tools (no graph) | Multi-step reasoning |
| **Full System** | GraphRAG + ReAct combined | All query types |

---

## Prerequisites

### From Phase 5

- [ ] Multi-hop QA results saved
- [ ] All 4 systems functional
- [ ] Test question bank created

### Python Dependencies

```python
# Add to requirements.txt
ragas>=0.1.0            # Evaluation framework
scipy>=1.10.0           # Statistical tests
matplotlib>=3.7.0       # Visualizations
seaborn>=0.12.0         # Statistical plots
plotly>=5.14.0          # Interactive charts
jinja2>=3.1.0           # Report templates
```

---

## Step 1: RAGAS Evaluation

### What We're Doing

Using the RAGAS (Retrieval Augmented Generation Assessment) framework to measure quality.

### Why

- **Standard Framework**: RAGAS is the industry standard for RAG evaluation
- **Four Metrics**: Covers all aspects of RAG quality
- **Reproducible**: Automated, no human annotation needed

### The Four RAGAS Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAGAS METRICS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. FAITHFULNESS (0-1)                                          │
│     "Is the answer based on the retrieved context?"             │
│     High = Answer doesn't hallucinate                           │
│     Low = Answer makes things up                                │
│                                                                  │
│  2. ANSWER RELEVANCY (0-1)                                      │
│     "Does the answer address the question?"                     │
│     High = Answer is on-topic                                   │
│     Low = Answer is tangential                                  │
│                                                                  │
│  3. CONTEXT PRECISION (0-1)                                     │
│     "Is the retrieved context relevant?"                        │
│     High = Retrieved good chunks                                │
│     Low = Retrieved irrelevant chunks                           │
│                                                                  │
│  4. CONTEXT RECALL (0-1)                                        │
│     "Did we retrieve all needed information?"                   │
│     High = Got all necessary context                            │
│     Low = Missed important context                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How Each Metric is Calculated

```python
# Faithfulness
# Checks if answer claims are supported by context

claims = extract_claims(answer)
supported = 0
for claim in claims:
    if is_supported_by_context(claim, context):
        supported += 1
faithfulness = supported / len(claims)

# Example:
# Context: "John works at Microsoft"
# Answer: "John works at Microsoft and loves his job"
# Claims: ["John works at Microsoft", "John loves his job"]
# Supported: 1/2 = 0.5 faithfulness (second claim not in context)
```

```python
# Answer Relevancy
# Generates questions from answer, checks if similar to original

generated_questions = generate_questions(answer)
similarities = [
    cosine_similarity(original_question, gen_q)
    for gen_q in generated_questions
]
answer_relevancy = mean(similarities)

# Example:
# Original: "Who works at Microsoft?"
# Answer: "John works at Microsoft"
# Generated: "Who works at Microsoft?" "Where does John work?"
# Both similar → High relevancy
```

### Instructions

1. **Run the RAGAS Evaluation Notebook**

   ```
   notebooks/06_evaluation/01_ragas_evaluation.py
   ```

2. **Preparing Evaluation Data**

   ```python
   from src.evaluation.ragas_evaluator import EvaluationSample

   # Each evaluation sample needs:
   samples = []

   for qa_result in qa_results:
       sample = EvaluationSample(
           question=qa_result["question"],
           answer=qa_result["answer"],
           contexts=qa_result["retrieved_contexts"],  # List of strings
           ground_truth=qa_result.get("expected_answer")  # Optional
       )
       samples.append(sample)
   ```

3. **Running RAGAS Evaluation**

   ```python
   from src.evaluation.ragas_evaluator import RAGASEvaluator

   evaluator = RAGASEvaluator(
       model="gpt-4o",
       metrics=["faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]
   )

   # Evaluate each system
   systems = ["baseline", "graphrag", "react", "full_system"]
   all_results = {}

   for system in systems:
       samples = load_samples(system)
       results = evaluator.evaluate(samples)
       all_results[system] = results

       # Results contain per-sample scores and aggregates
       print(f"{system} - Avg Faithfulness: {results.faithfulness.mean():.3f}")
   ```

4. **Understanding Results**

   ```python
   # Results structure

   results = evaluator.evaluate(samples)

   # Per-sample scores
   for i, sample_result in enumerate(results.sample_results):
       print(f"Q{i}: Faithfulness={sample_result.faithfulness:.2f}, "
             f"Relevancy={sample_result.answer_relevancy:.2f}")

   # Aggregate scores
   print(f"Mean Faithfulness: {results.aggregated.faithfulness_mean:.3f}")
   print(f"Std Faithfulness: {results.aggregated.faithfulness_std:.3f}")
   print(f"Mean Answer Relevancy: {results.aggregated.answer_relevancy_mean:.3f}")
   ```

### Expected Output

```
RAGAS Evaluation Results:
├── Per-sample scores for 50 questions × 4 systems = 200 evaluations
├── Aggregate scores per system
└── Score distributions

Gold Layer:
├── /mnt/datalake/gold/ragas_sample_results/
│   └── Individual sample scores
├── /mnt/datalake/gold/ragas_aggregate_results/
│   └── System-level aggregates
└── /mnt/datalake/gold/ragas_distributions/
    └── Score distributions for plotting

Example Results (Expected Pattern):
┌──────────────┬─────────────┬──────────┬───────────┬──────────────┐
│ System       │ Faithfulness│ Relevancy│ Precision │ Recall       │
├──────────────┼─────────────┼──────────┼───────────┼──────────────┤
│ Baseline     │ 0.78        │ 0.75     │ 0.65      │ 0.60         │
│ GraphRAG     │ 0.82        │ 0.80     │ 0.75      │ 0.70         │
│ ReAct        │ 0.85        │ 0.82     │ 0.70      │ 0.72         │
│ Full System  │ 0.88        │ 0.85     │ 0.80      │ 0.78         │
└──────────────┴─────────────┴──────────┴───────────┴──────────────┘
```

### Cost Estimation

```
RAGAS uses LLM calls to compute metrics:
- Faithfulness: ~500 tokens/sample
- Answer Relevancy: ~500 tokens/sample
- Context Precision: ~300 tokens/sample
- Context Recall: ~300 tokens/sample

Per sample: ~1,600 tokens
Total: 200 samples × 1,600 = 320,000 tokens
Cost: ~$3-5
```

---

## Step 2: Statistical Analysis

### What We're Doing

Performing rigorous statistical tests to prove improvements are significant.

### Why

- **Scientific Standards**: "It looks better" isn't enough for a thesis
- **p-values**: Prove differences aren't due to random chance
- **Effect Sizes**: Show how big the improvements are

### Statistical Tests We Use

```
1. PAIRED T-TEST
   Purpose: Compare two systems on same questions
   Use when: Comparing Full System vs Baseline
   Result: p-value (significant if p < 0.05)

2. EFFECT SIZE (Cohen's d)
   Purpose: Measure magnitude of improvement
   Interpretation:
   - d = 0.2: Small effect
   - d = 0.5: Medium effect
   - d = 0.8: Large effect

3. CONFIDENCE INTERVALS
   Purpose: Range of likely true difference
   Example: "Full System is 5-15% better (95% CI)"
```

### Instructions

1. **Run the Comparative Analysis Notebook**

   ```
   notebooks/06_evaluation/02_comparative_analysis.py
   ```

2. **Performing Paired T-Tests**

   ```python
   from src.evaluation.comparative_analysis import ComparativeAnalyzer
   from scipy import stats

   analyzer = ComparativeAnalyzer()

   # Compare Full System vs Baseline
   baseline_scores = ragas_results["baseline"]["faithfulness"]
   full_scores = ragas_results["full_system"]["faithfulness"]

   # Paired t-test (same questions, different systems)
   t_stat, p_value = stats.ttest_rel(full_scores, baseline_scores)

   print(f"t-statistic: {t_stat:.3f}")
   print(f"p-value: {p_value:.4f}")

   if p_value < 0.05:
       print("✓ Statistically significant improvement!")
   else:
       print("✗ Not statistically significant")
   ```

3. **Calculating Effect Sizes**

   ```python
   import numpy as np

   def cohens_d(group1, group2):
       """Calculate Cohen's d effect size."""
       n1, n2 = len(group1), len(group2)
       var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
       pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
       return (np.mean(group1) - np.mean(group2)) / pooled_std

   d = cohens_d(full_scores, baseline_scores)
   print(f"Cohen's d: {d:.3f}")

   # Interpretation
   if abs(d) < 0.2:
       print("Negligible effect")
   elif abs(d) < 0.5:
       print("Small effect")
   elif abs(d) < 0.8:
       print("Medium effect")
   else:
       print("Large effect")
   ```

4. **Analyzing by Question Type**

   ```python
   # See which system works best for which question type

   question_types = ["1-hop", "2-hop", "3-hop", "global"]

   for q_type in question_types:
       # Filter to this question type
       type_results = filter_by_type(ragas_results, q_type)

       # Find best system for this type
       best_system = max(
           type_results.keys(),
           key=lambda s: type_results[s]["faithfulness_mean"]
       )

       print(f"{q_type}: Best system is {best_system}")
   ```

5. **Building Comparison Matrix**

   ```python
   # Pairwise comparisons between all systems

   systems = ["baseline", "graphrag", "react", "full_system"]
   comparison_matrix = {}

   for sys1 in systems:
       for sys2 in systems:
           if sys1 != sys2:
               comparison = analyzer.compare_systems(
                   system1_name=sys1,
                   system2_name=sys2,
                   system1_results=ragas_results[sys1],
                   system2_results=ragas_results[sys2]
               )
               comparison_matrix[(sys1, sys2)] = comparison

   # Results include:
   # - p_value for each metric
   # - effect_size for each metric
   # - confidence_intervals
   # - is_significant (True/False)
   ```

### Expected Output

```
Statistical Analysis Results:
├── Pairwise comparisons: 6 pairs × 4 metrics = 24 tests
├── Effect sizes for all comparisons
├── Significance indicators
└── Question type breakdown

Gold Layer:
├── /mnt/datalake/gold/statistical_tests/
│   └── T-test results
├── /mnt/datalake/gold/effect_sizes/
│   └── Cohen's d values
└── /mnt/datalake/gold/question_type_analysis/
    └── Performance by question type

Example Statistical Results:
┌─────────────────────────────────┬───────────┬──────────┬─────────────┐
│ Comparison                      │ p-value   │ Cohen's d│ Significant │
├─────────────────────────────────┼───────────┼──────────┼─────────────┤
│ Full System vs Baseline         │ 0.001     │ 0.82     │ Yes ***     │
│ Full System vs GraphRAG         │ 0.023     │ 0.45     │ Yes *       │
│ Full System vs ReAct            │ 0.045     │ 0.38     │ Yes *       │
│ GraphRAG vs Baseline            │ 0.008     │ 0.58     │ Yes **      │
│ ReAct vs Baseline               │ 0.012     │ 0.52     │ Yes *       │
│ GraphRAG vs ReAct               │ 0.234     │ 0.21     │ No          │
└─────────────────────────────────┴───────────┴──────────┴─────────────┘

Performance by Question Type:
┌─────────────┬──────────┬──────────┬─────────┬─────────────┐
│ Question    │ Baseline │ GraphRAG │ ReAct   │ Full System │
├─────────────┼──────────┼──────────┼─────────┼─────────────┤
│ 1-hop       │ 0.82     │ 0.84     │ 0.83    │ 0.85        │
│ 2-hop       │ 0.65     │ 0.78     │ 0.75    │ 0.85        │
│ 3-hop       │ 0.45     │ 0.68     │ 0.75    │ 0.82        │
│ Global      │ 0.55     │ 0.80     │ 0.60    │ 0.78        │
└─────────────┴──────────┴──────────┴─────────┴─────────────┘
```

---

## Step 3: Generate Final Report

### What We're Doing

Creating publication-ready reports with visualizations for the thesis.

### Why

- **Thesis Deliverable**: Need professional report document
- **Visual Evidence**: Charts communicate findings better than tables
- **Reproducibility**: All findings in one document

### Report Contents

```
1. EXECUTIVE SUMMARY
   - Key findings in 1 paragraph
   - Best performing system
   - Main contribution

2. METHODOLOGY
   - Systems compared
   - Evaluation metrics
   - Test question details

3. RESULTS
   - RAGAS scores table
   - Statistical significance table
   - Performance by question type

4. VISUALIZATIONS
   - Bar chart: System comparison
   - Radar chart: Multi-metric view
   - Heatmap: Question type performance
   - Box plots: Score distributions

5. DISCUSSION
   - Why Full System performs best
   - Where each system excels
   - Limitations

6. CONCLUSIONS
   - Thesis hypothesis confirmed/rejected
   - Recommendations
```

### Instructions

1. **Run the Final Report Notebook**

   ```
   notebooks/06_evaluation/03_final_report.py
   ```

2. **Creating Visualizations**

   ```python
   from src.evaluation.report_generator import ReportGenerator
   import matplotlib.pyplot as plt
   import seaborn as sns

   generator = ReportGenerator(output_dir="/mnt/datalake/gold/reports")

   # 1. Bar Chart: System Comparison
   def create_comparison_bar_chart(results):
       metrics = ["faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]
       systems = ["Baseline", "GraphRAG", "ReAct", "Full System"]

       fig, ax = plt.subplots(figsize=(12, 6))
       x = np.arange(len(metrics))
       width = 0.2

       for i, system in enumerate(systems):
           scores = [results[system][m] for m in metrics]
           ax.bar(x + i*width, scores, width, label=system)

       ax.set_ylabel('Score')
       ax.set_title('RAGAS Metrics by System')
       ax.set_xticks(x + width * 1.5)
       ax.set_xticklabels(metrics)
       ax.legend()
       ax.set_ylim(0, 1)

       return fig
   ```

   ```python
   # 2. Radar Chart: Multi-metric View
   def create_radar_chart(results):
       categories = ["Faithfulness", "Relevancy",
                     "Precision", "Recall"]
       N = len(categories)

       angles = [n / float(N) * 2 * np.pi for n in range(N)]
       angles += angles[:1]  # Complete the circle

       fig, ax = plt.subplots(figsize=(8, 8),
                              subplot_kw=dict(polar=True))

       for system, scores in results.items():
           values = list(scores.values())
           values += values[:1]
           ax.plot(angles, values, 'o-', label=system)
           ax.fill(angles, values, alpha=0.25)

       ax.set_xticks(angles[:-1])
       ax.set_xticklabels(categories)
       ax.legend(loc='upper right')

       return fig
   ```

   ```python
   # 3. Heatmap: Question Type Performance
   def create_heatmap(results):
       question_types = ["1-hop", "2-hop", "3-hop", "global"]
       systems = ["Baseline", "GraphRAG", "ReAct", "Full System"]

       data = np.array([
           [results[s][q] for q in question_types]
           for s in systems
       ])

       fig, ax = plt.subplots(figsize=(10, 6))
       sns.heatmap(data, annot=True, fmt=".2f",
                   xticklabels=question_types,
                   yticklabels=systems,
                   cmap="RdYlGn", vmin=0, vmax=1)
       ax.set_title("Faithfulness by Question Type")

       return fig
   ```

3. **Generating the Report**

   ```python
   from src.evaluation.report_generator import ReportGenerator

   generator = ReportGenerator(
       output_dir="/mnt/datalake/gold/evaluation_reports"
   )

   # Generate all formats
   generator.generate_report(
       ragas_results=ragas_results,
       statistical_results=statistical_results,
       question_analysis=question_analysis,
       formats=["html", "markdown", "latex"]  # All three
   )

   # Output:
   # - evaluation_report.html  (interactive, for viewing)
   # - evaluation_report.md    (for GitHub/README)
   # - evaluation_report.tex   (for thesis LaTeX)
   ```

4. **Report Structure**

   ```python
   # The report includes:

   report = {
       "executive_summary": "...",
       "methodology": {
           "systems": [...],
           "metrics": [...],
           "test_data": {...}
       },
       "results": {
           "ragas_scores": {...},
           "statistical_tests": {...},
           "question_analysis": {...}
       },
       "visualizations": {
           "comparison_chart": "path/to/chart.png",
           "radar_chart": "path/to/radar.png",
           "heatmap": "path/to/heatmap.png",
           "box_plots": "path/to/boxes.png"
       },
       "discussion": "...",
       "conclusions": "...",
       "recommendations": [...]
   }
   ```

### Expected Output

```
Final Reports Generated:
├── /mnt/datalake/gold/evaluation_reports/
│   ├── evaluation_report.html     (7 pages)
│   ├── evaluation_report.md       (for README)
│   ├── evaluation_report.tex      (for thesis)
│   └── figures/
│       ├── system_comparison.png
│       ├── radar_chart.png
│       ├── question_heatmap.png
│       ├── score_distributions.png
│       └── statistical_significance.png
└── /mnt/datalake/gold/raw_data/
    └── All raw data for reproducibility
```

### Sample Report Sections

**Executive Summary:**
```
This study evaluated four knowledge retrieval systems on 50 multi-hop
questions from enterprise data. The Full System (GraphRAG + ReAct)
achieved the highest scores across all RAGAS metrics, with statistically
significant improvements over the baseline (p < 0.001, Cohen's d = 0.82).
The most dramatic improvements were seen in multi-hop reasoning tasks,
where the Full System achieved 82% faithfulness compared to 45% for
the baseline.
```

**Conclusion:**
```
The thesis hypothesis that combining knowledge graphs with ReAct agents
improves enterprise knowledge retrieval is supported by the evidence.
The Full System is recommended for:
- Complex multi-hop questions (85% vs 45% baseline)
- Relationship queries (80% vs 55% baseline)
- Global theme questions (78% vs 55% baseline)

The baseline RAG remains sufficient for simple factual queries
(85% vs 82% full system) where the additional complexity is not justified.
```

---

## Phase 6 Checklist

Before finalizing thesis, verify:

- [ ] RAGAS Evaluation Complete
  - [ ] All 4 systems evaluated
  - [ ] All 4 metrics calculated
  - [ ] Per-sample and aggregate scores saved

- [ ] Statistical Analysis Complete
  - [ ] Paired t-tests run
  - [ ] Effect sizes calculated
  - [ ] Significance determined
  - [ ] Question type analysis done

- [ ] Reports Generated
  - [ ] HTML report viewable
  - [ ] Markdown for GitHub
  - [ ] LaTeX for thesis
  - [ ] All visualizations included

- [ ] Data Archived
  - [ ] Raw results in Delta Lake
  - [ ] Figures saved as PNG
  - [ ] Code reproducible

---

## Verification Queries

```python
# Verify RAGAS results
ragas_df = spark.read.format("delta").load(
    "/mnt/datalake/gold/ragas_aggregate_results"
)
ragas_df.show()

# Verify statistical tests
stats_df = spark.read.format("delta").load(
    "/mnt/datalake/gold/statistical_tests"
)
stats_df.show()

# Check report files exist
import os
report_dir = "/dbfs/mnt/datalake/gold/evaluation_reports"
print(os.listdir(report_dir))

# Validate HTML report
with open(f"{report_dir}/evaluation_report.html", "r") as f:
    content = f.read()
    print(f"Report size: {len(content)} characters")
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| RAGAS scores all low | Poor retrieval | Check Phase 3 indexing |
| Not statistically significant | Too few samples | Add more test questions |
| Visualizations not rendering | Missing backend | Install `matplotlib` correctly |
| LaTeX errors | Special characters | Escape underscores, ampersands |
| Large effect but no significance | High variance | Need more consistent results |

---

## Final Cost Summary (All Phases)

| Phase | Component | Estimated Cost |
|-------|-----------|---------------|
| Phase 1 | Data ingestion | $0 (storage only) |
| Phase 2 | Summarization (GPT-4o) | $100-200 |
| Phase 3 | Embeddings | $50-100 |
| Phase 4 | Entity extraction + summaries | $600-700 |
| Phase 5 | Agent development | $50-100 |
| Phase 6 | RAGAS evaluation | $5-10 |
| **Total** | | **$800-1100** |

Plus Azure services:
- Databricks: ~$200-400/month
- AI Search: ~$75/month
- Cosmos DB: ~$50/month (optional)
- Storage: ~$20/month

---

## Thesis Submission Checklist

- [ ] All notebooks run successfully
- [ ] Delta Lake tables populated
- [ ] Azure AI Search indexes working
- [ ] Evaluation report complete
- [ ] Statistical significance demonstrated
- [ ] All visualizations generated
- [ ] Code documented
- [ ] README updated

---

## Congratulations!

You have completed the GraphRAG + ReAct Knowledge Retrieval System!

Your thesis deliverables:
1. **Working System**: Enterprise knowledge retrieval with multi-hop QA
2. **Evaluation Report**: RAGAS metrics proving improvements
3. **Statistical Evidence**: Significance tests supporting hypothesis
4. **Visualizations**: Publication-ready charts and figures

---

*Phase 6 Complete! Your thesis system is ready for submission.*

---

*GraphRAG + ReAct Knowledge Retrieval System*
*KU Leuven Master Thesis - Muhammad Rafiq*
*February 2026*
