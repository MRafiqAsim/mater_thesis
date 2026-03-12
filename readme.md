# SimpleRAG Pipeline

Enterprise email knowledge extraction pipeline using Bronze/Silver/Gold medallion architecture.

## Architecture

```
PST Files → Bronze (raw extraction) → Silver (chunk → anonymize → summarize) → Gold (embed + index) → Query
```

### Silver Staged Folders

```
silver/
├── chunks/                  ← Stage 1: Clean text chunks (disclaimer-removed, NOT anonymized)
├── chunks_anonymized/       ← Stage 2: PII-anonymized chunks
├── chunks_summarized/       ← Stage 3: Anonymized + summarized (final output)
└── ocr/                     ← OCR outputs
```

Each stage is independently re-runnable. Deleting `chunks_anonymized/` and re-running Stage 2 preserves the original clean chunks in `chunks/`.

---

## Files to Transfer to Windows

### Source Code (required)

```
src/
└── simplerag/
    ├── __init__.py
    ├── app.py                        # Gradio web UI
    ├── ask.py                        # CLI query interface
    ├── run_pipeline.py               # Main pipeline runner
    ├── bronze/
    │   ├── __init__.py
    │   ├── ingestion.py              # PST/email extraction
    │   ├── libpst_extractor.py       # libpst-based extractor
    │   └── message_index.py          # Cross-PST deduplication
    ├── silver/
    │   ├── __init__.py
    │   ├── processor.py              # 3-stage Silver processor
    │   ├── disclaimer_remover.py     # Email boilerplate removal
    │   └── thread_grouper.py         # Email thread grouping
    ├── gold/
    │   ├── __init__.py
    │   └── rag_retriever.py          # Vector search + RAG
    └── utils/
        ├── __init__.py
        ├── config.py                 # Configuration
        └── lineage.py                # Lineage tracking
```

### Configuration (required)

```
.env                    ← Your Azure OpenAI credentials (create from .env.template)
.env.template           ← Template with all env vars
```

### Data (transfer what you need)

```
data/simplerag/
├── bronze/             ← 1.3 GB - Raw extracted emails (needed to re-run Silver)
├── silver/
│   ├── chunks_summarized/  ← 2.7 MB - Final Silver output (needed for Gold)
│   ├── chunks/             ← Empty (populated by Stage 1)
│   └── chunks_anonymized/  ← Empty (populated by Stage 2)
├── gold/               ← 6 MB - Embeddings + index (needed for queries)
└── source/             ← PST files (needed to re-run Bronze)
```

**Minimum transfer for querying only**: `gold/` folder (6 MB) + source code + `.env`
**Minimum for re-indexing**: add `silver/chunks_summarized/` (2.7 MB)
**Full pipeline re-run**: add `bronze/` (1.3 GB) or `source/` (PST files)

---

## Windows Setup (WSL)

### 1. Install Prerequisites

```bash
# Update WSL
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3-pip -y

# Install libpst (for PST extraction)
sudo apt install pst-utils -y
```

### 2. Clone / Copy Project

```bash
# Option A: Clone from GitHub
git clone https://github.com/MRafiqAsim/mater_thesis.git
cd mater_thesis

# Option B: Copy from USB/network
# Copy the project folder into WSL, e.g. /home/rafiq/mater_thesis/
```

### 3. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install SimpleRAG dependencies only (lightweight)
pip install openai httpx numpy python-dotenv gradio
```

The full `requirements.txt` includes GraphRAG/PathRAG dependencies you don't need for SimpleRAG.

### 5. Configure Environment

```bash
# Copy template and fill in your credentials
cp .env.template .env
nano .env
```

Required variables for SimpleRAG:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 6. Copy Data

```bash
# Create data directory
mkdir -p data/simplerag

# Copy your data folders (from USB, scp, etc.)
# At minimum, copy gold/ for querying:
cp -r /mnt/c/Users/YourName/simplerag_data/gold data/simplerag/

# For full pipeline, also copy bronze/ and silver/
cp -r /mnt/c/Users/YourName/simplerag_data/bronze data/simplerag/
cp -r /mnt/c/Users/YourName/simplerag_data/silver data/simplerag/
```

---

## Running the Pipeline

### Full Pipeline

```bash
# Run everything: Bronze → Silver (all 3 stages) → Gold
python -m src.simplerag.run_pipeline --all

# Run individual layers
python -m src.simplerag.run_pipeline --bronze
python -m src.simplerag.run_pipeline --silver
python -m src.simplerag.run_pipeline --gold
```

### Silver Stages (Individual)

```bash
# Stage 1: Extract text, remove disclaimers, chunk (NO anonymization)
python -m src.simplerag.run_pipeline --silver-chunks

# Stage 2: Anonymize PII in chunks
python -m src.simplerag.run_pipeline --silver-anonymize

# Stage 3: Generate summaries for anonymized chunks
python -m src.simplerag.run_pipeline --silver-summarize
```

### Query

```bash
# Single question
python -m src.simplerag.run_pipeline --query "What projects are discussed?"

# With detailed sources
python -m src.simplerag.ask "What projects are discussed?" -v

# Interactive CLI mode
python -m src.simplerag.ask -i

# Web UI (Gradio) at http://localhost:7860
python -m src.simplerag.app
```

### Index Stats

```bash
python -m src.simplerag.ask --stats
```

---

## Re-run, Resume, and Reset

### Resume (skip already processed records)

All stages automatically skip records that already exist in their output folder. Just run the command again:

```bash
# If interrupted during Silver Stage 2, re-run — it skips already anonymized records
python -m src.simplerag.run_pipeline --silver-anonymize

# Same for Gold — skips already embedded chunks
python -m src.simplerag.run_pipeline --gold
```

### Re-run a Specific Stage with Different Settings

```bash
# Example: Re-run anonymization with a different prompt
# 1. Delete the stage output you want to redo
rm -rf data/simplerag/silver/chunks_anonymized/*.json

# 2. (Optional) Edit the anonymization prompt in src/simplerag/silver/processor.py

# 3. Re-run Stage 2 — it reads from chunks/ and writes fresh to chunks_anonymized/
python -m src.simplerag.run_pipeline --silver-anonymize

# 4. Re-run Stage 3 (summaries depend on anonymized text)
rm -rf data/simplerag/silver/chunks_summarized/*.json
python -m src.simplerag.run_pipeline --silver-summarize
```

### Re-run Summarization Only

```bash
# Delete existing summaries
rm -rf data/simplerag/silver/chunks_summarized/*.json

# Re-run — reads from chunks_anonymized/, generates new summaries
python -m src.simplerag.run_pipeline --silver-summarize
```

### Rebuild Gold Index from Scratch

```bash
# Option A: Delete and re-index
rm -rf data/simplerag/gold/index/* data/simplerag/gold/embeddings/*
python -m src.simplerag.run_pipeline --gold

# Option B: Use rebuild in Python
python -c "
import asyncio
from src.simplerag.run_pipeline import SimpleRAGPipeline
async def rebuild():
    p = SimpleRAGPipeline()
    await p.gold.rebuild_index()
asyncio.run(rebuild())
"
```

### Re-run Full Pipeline from Scratch

```bash
# Delete all Silver and Gold data
rm -rf data/simplerag/silver/chunks/*
rm -rf data/simplerag/silver/chunks_anonymized/*
rm -rf data/simplerag/silver/chunks_summarized/*
rm -rf data/simplerag/gold/index/*
rm -rf data/simplerag/gold/embeddings/*

# Re-run everything
python -m src.simplerag.run_pipeline --silver
python -m src.simplerag.run_pipeline --gold
```

### Stop and Resume

The pipeline processes records one-by-one. To stop, press `Ctrl+C`. To resume, just re-run the same command — it automatically skips completed records.

```bash
# Start processing (will take a while for 249 records)
python -m src.simplerag.run_pipeline --silver

# Press Ctrl+C to stop at any time

# Resume later — picks up where it left off
python -m src.simplerag.run_pipeline --silver
```

---

## Stage Dependency Chain

```
Bronze records ──→ --silver-chunks ──→ --silver-anonymize ──→ --silver-summarize ──→ --gold
                   (silver/chunks/)    (chunks_anonymized/)   (chunks_summarized/)   (gold/index/)
```

- You can re-run any stage without affecting upstream stages
- Downstream stages must be re-run if you change an upstream stage
- `--gold` always reads from `silver/chunks_summarized/`
- Summaries are embedded for search; actual chunk text is used for LLM response generation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: openai` | Activate venv: `source .venv/bin/activate` |
| `AZURE_OPENAI_ENDPOINT not set` | Check `.env` file has correct credentials |
| Empty query results | Run `python -m src.simplerag.ask --stats` to check index |
| Gold index out of date | Delete `gold/index/` and `gold/embeddings/`, re-run `--gold` |
| Want to change anonymization | Delete `chunks_anonymized/` + `chunks_summarized/`, re-run stages 2 & 3 |
| WSL can't find Python | Install: `sudo apt install python3.12 python3.12-venv` |
| Gradio port in use | Kill: `lsof -ti:7860 | xargs kill` (Mac) or `fuser -k 7860/tcp` (WSL) |

## Attachment Classification Strategy

Attachments are classified as **knowledge** (reports, manuals, policies) or **transactional** (invoices, spreadsheets, data exports) in the **Bronze layer** using a multi-signal weighted scoring model. This determines how Silver processes them — knowledge docs get full processing; transactional docs are token-capped.

### Why Bronze, Not Silver?

Classification answers "what is this data?" — that's an ingestion concern. Silver should consume the label, not compute it. Moving classification to Bronze also means the result is cached alongside the extracted text, so re-running Silver doesn't re-classify.

### Four Signals

| Signal | Weight | What It Looks At |
|--------|--------|------------------|
| **Content patterns** | 0.45 | First ~2000 chars of extracted text. Looks for tab-separated columns, currency amounts, financial headers (transactional) vs. section numbering, table of contents, prose density, legal language (knowledge). |
| **Email body context** | 0.20 | Keywords near "attach" mentions in the parent email. "Attached invoice" → transactional. "Attached report" → knowledge. |
| **Document structure** | 0.20 | Has tables? Doc type (xlsx vs docx)? Page count? High table density? |
| **Filename/extension** | 0.15 | `.xlsx` → strongly transactional. `.docx` → leans knowledge. `.pdf` → neutral. Keyword scan on filename stem. |

### How Scoring Works

Each signal produces a score from **-1.0** (transactional) to **+1.0** (knowledge). The weighted sum determines classification:

```
weighted_sum = 0.45*content + 0.20*email_context + 0.20*structure + 0.15*filename

if weighted_sum >= 0  →  "knowledge"
if weighted_sum <  0  →  "transactional"
```

**Confidence** maps the magnitude through an exponential curve:

```
confidence = 0.5 + 0.5 * (1 - e^(-3 * |weighted_sum|))

|weighted_sum| = 0.0  →  confidence = 0.50  (coin flip, no signals)
|weighted_sum| = 0.33 →  confidence = 0.82
|weighted_sum| = 0.50 →  confidence = 0.89
|weighted_sum| = 1.0  →  confidence = 0.98
```

### Example Classifications

| Attachment | Result | Key Signals | Confidence |
|------------|--------|-------------|------------|
| TeamFund Receivables.xlsx | transactional | tab rows + .xlsx extension | ~0.94 |
| GT Invoices - Exported Data.xlsx | transactional | tab rows, $amounts, "Invoice Number" header | ~0.96 |
| SL Service Rules - Main Book.docx | knowledge | ToC, section headers, policy language, prose | ~0.95 |
| WFG - LoanDepot Documents.docx | transactional | table structure overrides .docx extension | ~0.69 |
| 123.pdf (invoice scan) | transactional | financial headers, amounts | ~0.75 |
| unknown.pdf (empty/scanned) | knowledge (default) | no signals → neutral → default knowledge | ~0.50 |

### Audit Trail

Every classification stores the full signal breakdown in `classification_signals` — you can inspect exactly why any attachment was classified the way it was by reading the cache JSON in `data/bronze/attachments_cache/`.

### Silver Output Structure

Classified attachments are routed into subdirectories:

```
silver/
├── email_chunks/               <- email body (single emails)
├── email_summaries/            <- per-email summaries
├── thread_chunks/              <- email body (threaded emails)
├── attachment_chunks/
│   ├── knowledge/              <- full-processed knowledge docs
│   │   └── att_{id}_{n}.json
│   └── transactional/          <- token-capped transactional docs
│       └── att_{id}_{n}.json
├── thread_summaries/
├── pii_mappings/
└── metadata/
```

### Source Code

- Classifier: `src/ingestion/attachment_classifier.py`
- Integration: `src/ingestion/attachment_processor.py` (Bronze)
- Consumer: `src/anonymization/thread_aware_processor.py` (Silver)

---

## Learning Resources

- [Azure AI Search Training](https://learn.microsoft.com/en-us/training/modules/create-azure-cognitive-search-solution/3-search-components)
- [What is Azure AI Search?](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search?tabs=indexing%2Cquickstarts)
