# Data Lake Architecture - Medallion Pattern

The **Bronze, Silver, Gold** layers are the **Medallion Architecture** pattern for organizing data in Delta Lake.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAKE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   RAW DATA          CLEANED DATA         BUSINESS-READY DATA       │
│      │                   │                       │                  │
│      ▼                   ▼                       ▼                  │
│  ┌────────┐         ┌────────┐              ┌────────┐             │
│  │ BRONZE │   ───►  │ SILVER │    ───►      │  GOLD  │             │
│  └────────┘         └────────┘              └────────┘             │
│                                                                     │
│  "Land as-is"      "Clean & enrich"      "Ready for use"           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Bronze Layer (Raw Data)

**Path:** `/mnt/datalake/bronze/`

**What it is:** Raw data exactly as received - no changes.

### Tables in Bronze Layer

| Table | Contents |
|-------|----------|
| `bronze/emails/` | Raw PST email extracts (sender, recipient, body, attachments) |
| `bronze/documents/` | Raw document text (PDF, DOCX, XLSX, PPTX content) |
| `bronze/attachments/` | Email attachment files |
| `bronze/metadata/` | File metadata (size, date, path, type) |

### Characteristics

- ✅ Exact copy of source data
- ✅ No transformations applied
- ✅ Keeps original format/errors
- ✅ Used for reprocessing if needed

---

## Silver Layer (Cleaned & Enriched)

**Path:** `/mnt/datalake/silver/`

**What it is:** Cleaned, validated, and enriched data.

### Tables in Silver Layer

| Table | Contents |
|-------|----------|
| `silver/chunks/` | Semantically chunked text (512-1024 tokens each) |
| `silver/chunks_anonymized/` | Chunks with PII replaced |
| `silver/ner_results/` | Named entity recognition output |
| `silver/summaries/` | Chunk and document summaries |
| `silver/entities_raw/` | Raw extracted entities before deduplication |
| `silver/relationships_raw/` | Raw extracted relationships |
| `silver/extraction_metadata/` | Entity extraction statistics |

### Characteristics

- ✅ Data is cleaned and validated
- ✅ Duplicates removed
- ✅ Schema enforced
- ✅ Enriched with NER, language tags, etc.

---

## Gold Layer (Business-Ready)

**Path:** `/mnt/datalake/gold/`

**What it is:** Final, aggregated, ready-to-use data for applications.

### Tables in Gold Layer

| Table | Contents |
|-------|----------|
| `gold/entities/` | Deduplicated, normalized entities |
| `gold/relationships/` | Processed relationships with entity IDs |
| `gold/communities/` | Detected community clusters |
| `gold/community_summaries/` | GPT-4o generated community summaries |
| `gold/community_summaries_indexed/` | Summaries with embeddings |
| `gold/chunks_embedded/` | Chunks with vector embeddings |
| `gold/graph_statistics/` | Knowledge graph metrics |
| `gold/retriever_config/` | GraphRAG retriever settings |
| `gold/agent_config/` | ReAct agent configuration |
| `gold/qa_evaluation_results/` | Multi-hop QA test results |
| `gold/ragas_evaluation_results/` | RAGAS metric scores |
| `gold/comparative_analysis_results/` | System comparison data |
| `gold/evaluation_reports/` | Final reports and visualizations |

### Characteristics

- ✅ Ready for ML/AI applications
- ✅ Optimized for queries
- ✅ Aggregated and joined data
- ✅ Business metrics calculated

---

## Data Flow in the Project

```
PHASE 1-2: Data Ingestion & Processing
──────────────────────────────────────

  PST Files ──┐
  PDF Files ──┼──► BRONZE ──► Chunk ──► Clean ──► SILVER
  DOCX Files ─┤     (raw)     (split)   (NER,     (processed)
  Emails ─────┘                         PII)


PHASE 3-4: Indexing & GraphRAG
──────────────────────────────

  SILVER ──► Embed ──► Index ──► GOLD
  (chunks)   (vectors)  (search)  (ready for RAG)

  SILVER ──► Extract ──► Graph ──► Communities ──► GOLD
  (chunks)   (entities)  (build)   (detect)        (knowledge graph)


PHASE 5-6: Agent & Evaluation
─────────────────────────────

  GOLD ──► RAG System ──► Evaluate ──► GOLD
  (all)    (answer Qs)    (RAGAS)      (results)
```

---

## Simple Analogy

| Layer | Analogy | Example |
|-------|---------|---------|
| **Bronze** | Raw ingredients | Whole vegetables from the farm |
| **Silver** | Prepped ingredients | Washed, chopped, ready to cook |
| **Gold** | Finished dish | Cooked meal ready to serve |

---

## Why This Matters

1. **Reprocessing**: If Silver processing fails, Bronze still has original data
2. **Debugging**: Can trace issues back through layers
3. **Versioning**: Delta Lake keeps history at each layer
4. **Performance**: Gold is optimized for fast queries
5. **Governance**: Clear lineage from raw to final data

---

## Quick Commands to Check Your Data

```python
# Check Bronze layer
display(spark.read.format("delta").load("/mnt/datalake/bronze/documents"))

# Check Silver layer
display(spark.read.format("delta").load("/mnt/datalake/silver/chunks"))

# Check Gold layer
display(spark.read.format("delta").load("/mnt/datalake/gold/entities"))

# See all tables in a layer
dbutils.fs.ls("/mnt/datalake/gold/")
```

---

## Layer Summary Table

| Aspect | Bronze | Silver | Gold |
|--------|--------|--------|------|
| **Purpose** | Store raw data | Clean & enrich | Serve applications |
| **Data Quality** | As-is (may have errors) | Validated & cleaned | Business-ready |
| **Schema** | May vary | Enforced | Optimized |
| **Updates** | Append only | Transform from Bronze | Aggregate from Silver |
| **Users** | Data engineers | Data engineers | Data scientists, Apps |
| **Example** | Raw email text | Chunked + NER tagged | Entity with embeddings |

---

*Part of the GraphRAG + ReAct Knowledge Retrieval System*
*KU Leuven Master Thesis - Muhammad Rafiq*
