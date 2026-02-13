# Pipeline Setup & Execution Guide

A step-by-step guide to set up and run the anonymization and summarization pipeline.

## Table of Contents

1. [Project Directory Structure](#1-project-directory-structure)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Configuration](#4-configuration)
5. [Prepare Your Data](#5-prepare-your-data)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Evaluate Quality](#7-evaluate-quality)
8. [Understanding the Output](#8-understanding-the-output)
9. [Email Attachments Handling](#9-email-attachments-handling)
10. [Rerun Behavior & Incremental Processing](#10-rerun-behavior--incremental-processing)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Directory Structure

### Complete Project Structure

Create this directory structure to implement the pipeline:

```
mater_thesis/
│
├── .env                          # Environment variables (create from .env.template)
├── .env.template                 # Template for environment variables
├── requirements.txt              # Python dependencies
│
├── src/                          # SOURCE CODE
│   ├── __init__.py
│   ├── config.py                 # Central configuration (ProcessingMode, etc.)
│   │
│   ├── ingestion/                # Data ingestion module
│   │   ├── __init__.py
│   │   ├── pst_extractor.py      # Extract emails from PST files
│   │   ├── document_parser.py    # Parse PDF, DOCX, XLSX, etc.
│   │   ├── chunker.py            # Semantic text chunking
│   │   ├── language_detector.py  # Detect EN/NL language
│   │   └── bronze_loader.py      # Load data to Bronze layer
│   │
│   ├── anonymization/            # PII detection & anonymization
│   │   ├── __init__.py
│   │   ├── pii_detector.py       # Local PII detection (Presidio/spaCy)
│   │   ├── anonymizer.py         # Anonymization strategies
│   │   ├── openai_pii_detector.py    # OpenAI-based PII detection
│   │   ├── openai_summarizer.py      # OpenAI-based summarization
│   │   ├── unified_processor.py      # Unified processor (all modes)
│   │   ├── silver_processor.py       # Bronze → Silver processing
│   │   └── privacy_metrics.py        # K-anonymity, L-diversity, etc.
│   │
│   ├── conflict_handling/        # Temporal & conflict handling
│   │   ├── __init__.py
│   │   ├── models.py             # Data models
│   │   ├── temporal_extractor.py # Extract dates from text
│   │   ├── metadata_extractor.py # Extract dates from file metadata
│   │   ├── conflict_detector.py  # Detect contradictions
│   │   ├── temporal_decay.py     # Recency weighting
│   │   ├── entity_versioning.py  # Track entity changes
│   │   ├── pii_evaluation.py     # Evaluate PII detection accuracy
│   │   └── conflict_aware_retriever.py
│   │
│   ├── evaluation/               # Quality evaluation
│   │   ├── __init__.py
│   │   └── summarization_metrics.py  # ROUGE, faithfulness, etc.
│   │
│   ├── pipeline/                 # Pipeline entry points
│   │   ├── __init__.py
│   │   ├── run_ingestion.py      # Main pipeline script
│   │   ├── evaluate_anonymization.py  # PII evaluation
│   │   ├── evaluate_privacy.py        # Privacy metrics
│   │   ├── evaluate_quality.py        # Combined evaluation
│   │   └── test_processing_modes.py   # Test all modes
│   │
│   ├── graphrag/                 # GraphRAG (future)
│   │   ├── __init__.py
│   │   ├── entity_extraction.py
│   │   ├── graph_store.py
│   │   ├── community_detection.py
│   │   └── community_summarization.py
│   │
│   ├── agents/                   # ReAct agents (future)
│   │   ├── __init__.py
│   │   ├── react_agent.py
│   │   ├── graphrag_retriever.py
│   │   └── tools.py
│   │
│   └── retrieval/                # RAG retrieval (future)
│       ├── __init__.py
│       ├── vector_search.py
│       ├── rag_chain.py
│       └── metrics.py
│
├── data/                         # DATA DIRECTORIES
│   ├── input/                    # Your source files
│   │   ├── pst/                  # PST email archives
│   │   └── documents/            # PDF, DOCX, etc.
│   │
│   ├── bronze/                   # Raw extracted data
│   │   ├── emails/               # Extracted emails (JSON)
│   │   ├── documents/            # Parsed documents (JSON)
│   │   └── attachments/          # Email attachments
│   │
│   ├── silver/                   # Processed & anonymized
│   │   ├── chunks/               # Anonymized text chunks
│   │   └── metadata/             # Processing metadata
│   │
│   ├── gold/                     # Business-ready (future)
│   │   ├── embeddings/           # Vector embeddings
│   │   └── knowledge_graph/      # Entity relationships
│   │
│   └── evaluation/               # Evaluation data
│       └── ground_truth_sample.json
│
├── docs/                         # DOCUMENTATION
│   ├── PIPELINE_GUIDE.md         # This guide
│   └── phase_guides/
│       └── WithPathRagAndReact.md  # Architecture document
│
├── notebooks/                    # JUPYTER NOTEBOOKS (optional)
│   ├── 01_ingestion/
│   ├── 02_nlp_processing/
│   ├── 03_vector_index/
│   ├── 04_graphrag/
│   ├── 05_react_agent/
│   └── 06_evaluation/
│
└── tests/                        # TEST FILES
    └── __init__.py
```

### Create Directory Structure

Run these commands to create the full structure:

```bash
# Create main directories
mkdir -p src/{ingestion,anonymization,conflict_handling,evaluation,pipeline,graphrag,agents,retrieval}
mkdir -p src/conflict_handling/examples
mkdir -p data/{input/pst,input/documents,bronze/emails,bronze/documents,bronze/attachments}
mkdir -p data/{silver/chunks,silver/metadata,gold/embeddings,gold/knowledge_graph,evaluation}
mkdir -p docs/phase_guides
mkdir -p notebooks/{01_ingestion,02_nlp_processing,03_vector_index,04_graphrag,05_react_agent,06_evaluation}
mkdir -p tests

# Create __init__.py files
touch src/__init__.py
touch src/{ingestion,anonymization,conflict_handling,evaluation,pipeline,graphrag,agents,retrieval}/__init__.py
touch src/conflict_handling/examples/__init__.py
touch tests/__init__.py
```

### Key Files for Each Module

#### Ingestion Module (Bronze Layer)
| File | Purpose |
|------|---------|
| `pst_extractor.py` | Extract emails from PST using pypff |
| `document_parser.py` | Parse PDF/DOCX/XLSX using pypdf/docx |
| `chunker.py` | Split text into semantic chunks |
| `language_detector.py` | Detect EN/NL using langdetect |
| `bronze_loader.py` | Save raw data as JSON |

#### Anonymization Module (Silver Layer)
| File | Purpose |
|------|---------|
| `pii_detector.py` | Local detection (Presidio + spaCy + regex) |
| `openai_pii_detector.py` | OpenAI-based detection |
| `openai_summarizer.py` | OpenAI-based summarization |
| `unified_processor.py` | Switch between modes |
| `anonymizer.py` | Replace/mask/hash PII |
| `privacy_metrics.py` | K-anonymity, L-diversity, etc. |

#### Pipeline Module (Entry Points)
| File | Purpose |
|------|---------|
| `run_ingestion.py` | Main CLI for full pipeline |
| `evaluate_anonymization.py` | Test PII detection accuracy |
| `evaluate_privacy.py` | Calculate privacy metrics |
| `evaluate_quality.py` | Combined quality evaluation |

#### Configuration Files
| File | Purpose |
|------|---------|
| `src/config.py` | ProcessingMode, PipelineConfig |
| `.env.template` | Environment variable template |
| `.env` | Your actual settings (git-ignored) |

---

---

## 2. Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.12.x | Runtime environment |
| pip | Latest | Package management |
| Git | Latest | Version control |

### Check Python Version

```bash
python3 --version
# Should show: Python 3.12.x
```

If you don't have Python 3.12:
```bash
# macOS with Homebrew
brew install python@3.12

# Ubuntu/Debian
sudo apt install python3.12 python3.12-venv
```

---

## 3. Environment Setup

### Step 3.1: Clone/Navigate to Project

```bash
cd /path/to/mater_thesis
```

### Step 3.2: Create Virtual Environment

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify
python --version  # Should show 3.12.x
```

### Step 3.3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install presidio-analyzer presidio-anonymizer spacy tiktoken python-dateutil pyyaml openai

# Download spaCy language models
python -m spacy download en_core_web_lg
python -m spacy download nl_core_news_lg
```

### Step 3.4: Verify Installation

```bash
# Quick test
python src/pipeline/evaluate_anonymization.py --quick-test
```

Expected output:
```
======================================================================
QUICK TEST - PII Detection
======================================================================
Input (en): Contact John Smith at john.smith@email.com...
Detected: ['John Smith', 'john.smith@email.com', '+1 555-123-4567']
```

---

## 4. Configuration

### Step 4.1: Create Environment File

```bash
# Copy template
cp .env.template .env

# Edit with your settings
nano .env  # or use your preferred editor
```

### Step 4.2: Configure Processing Mode

Edit `.env` and set your preferred mode:

```bash
# Option 1: OpenAI (recommended - best quality)
PIPELINE_MODE=openai
OPENAI_API_KEY=sk-your-api-key-here

# Option 2: Local only (no API needed, no summarization)
PIPELINE_MODE=local

# Option 3: Hybrid (local + OpenAI for complex cases)
PIPELINE_MODE=hybrid
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 4.3: Set Environment Variables

```bash
# Load from .env file
export $(grep -v '^#' .env | xargs)

# Or set manually
export OPENAI_API_KEY='sk-your-api-key-here'
export PIPELINE_MODE='openai'
```

### Processing Modes Comparison

| Mode | PII Detection | Summarization | API Cost | Speed |
|------|--------------|---------------|----------|-------|
| `openai` | OpenAI GPT-4o | Yes | $$$ | Slower |
| `local` | Presidio/spaCy | No | Free | Fast |
| `hybrid` | Local + OpenAI | Yes | $$ | Medium |

---

## 5. Prepare Your Data

### Step 5.1: Create Directory Structure

```bash
mkdir -p data/input/pst
mkdir -p data/input/documents
mkdir -p data/bronze
mkdir -p data/silver
mkdir -p data/gold
```

### Step 5.2: Add Your Files

```
data/
├── input/
│   ├── pst/                    ← Put PST files here
│   │   └── your_emails.pst
│   │
│   └── documents/              ← Put documents here
│       ├── report.pdf
│       ├── notes.docx
│       ├── data.xlsx
│       └── manual.txt
```

### Supported File Types

| Category | Extensions |
|----------|------------|
| **Email Archives** | `.pst` |
| **Documents** | `.pdf`, `.docx`, `.doc`, `.txt`, `.rtf` |
| **Spreadsheets** | `.xlsx`, `.xls` |
| **Presentations** | `.pptx`, `.ppt` |
| **Web/Email** | `.html`, `.eml`, `.msg` |

---

## 6. Run the Pipeline

### Step 6.1: Activate Environment

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
```

### Step 6.2: Process Documents Only

```bash
python src/pipeline/run_ingestion.py \
    --documents data/input/documents \
    --output data
```

### Step 6.3: Process PST Files Only

```bash
python src/pipeline/run_ingestion.py \
    --pst data/input/pst/your_emails.pst \
    --output data
```

### Step 6.4: Process Both (Full Pipeline)

```bash
python src/pipeline/run_ingestion.py \
    --pst data/input/pst/your_emails.pst \
    --documents data/input/documents \
    --output data
```

### Step 6.5: Process Existing Bronze → Silver

If you already have Bronze layer data:

```bash
python src/pipeline/run_ingestion.py \
    --bronze data/bronze \
    --silver data/silver
```

### Pipeline Options

| Option | Description |
|--------|-------------|
| `--pst PATH` | Path to PST file |
| `--documents PATH` | Path to documents folder |
| `--output PATH` | Output directory (default: ./data) |
| `--bronze PATH` | Use existing Bronze layer |
| `--silver PATH` | Silver layer output path |
| `--chunk-size N` | Chunk size in tokens (default: 512) |
| `--evaluate PATH` | Ground truth for evaluation |
| `--verbose` | Show detailed logs |

---

## 7. Evaluate Quality

### Step 7.1: Quick Demo (No Data Needed)

```bash
# Local mode (no API key needed)
PIPELINE_MODE=local python src/pipeline/evaluate_quality.py --demo

# OpenAI mode (with summarization)
python src/pipeline/evaluate_quality.py --demo
```

### Step 7.2: Evaluate Anonymization Accuracy

```bash
# Against provided sample ground truth
python src/pipeline/evaluate_quality.py \
    --anonymization \
    --ground-truth data/evaluation/ground_truth_sample.json

# Against your own ground truth
python src/pipeline/evaluate_quality.py \
    --anonymization \
    --ground-truth /path/to/your/ground_truth.json
```

### Step 7.3: Evaluate Summarization Quality

```bash
python src/pipeline/evaluate_quality.py \
    --summarization \
    --silver data/silver \
    --sample-size 20
```

### Step 7.4: Full Evaluation (Both)

```bash
python src/pipeline/evaluate_quality.py \
    --all \
    --ground-truth data/evaluation/ground_truth_sample.json \
    --silver data/silver \
    --output results/evaluation_report.json
```

### Step 7.5: Privacy Metrics

```bash
# Demo with sample data
python src/pipeline/evaluate_privacy.py --demo

# Evaluate Silver layer
python src/pipeline/evaluate_privacy.py \
    --silver data/silver \
    --output results/privacy_report.json
```

---

## 8. Understanding the Output

### Directory Structure After Processing

```
data/
├── input/                  # Your source files (unchanged)
│
├── bronze/                 # Raw extracted data
│   ├── emails/             # Extracted emails as JSON
│   │   └── email_001.json
│   ├── documents/          # Parsed documents as JSON
│   │   └── doc_001.json
│   ├── attachments/        # Email attachments
│   └── metadata.json       # Processing metadata
│
├── silver/                 # Processed & anonymized
│   ├── chunks/             # Text chunks with PII removed
│   │   └── chunk_001.json
│   └── metadata/           # Processing stats
│       └── processing_stats.json
│
├── gold/                   # Business-ready (future)
│   ├── embeddings/
│   └── knowledge_graph/
│
└── pipeline_stats.json     # Overall pipeline statistics
```

### Chunk JSON Structure

Each chunk in `silver/chunks/` contains:

```json
{
  "chunk_id": "chunk_001",
  "doc_id": "doc_001",
  "text_original": "Original text with PII...",
  "text_anonymized": "Text with [PERSON_1] and [EMAIL_1]...",
  "summary": "Brief summary of the content...",
  "pii_entities": [
    {
      "text": "John Smith",
      "type": "PERSON",
      "start": 10,
      "end": 20,
      "confidence": 0.95
    }
  ],
  "pii_count": 3,
  "language": "en",
  "token_count": 256
}
```

### Evaluation Metrics

#### Anonymization Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Precision | >90% | Correct detections / Total detections |
| Recall | >95% | Correct detections / Total actual PII |
| F1 Score | >92% | Harmonic mean of P & R |

#### Privacy Metrics
| Metric | Good Value | Description |
|--------|------------|-------------|
| K-anonymity | ≥5 | Records indistinguishable from k-1 others |
| L-diversity | ≥3 | Distinct sensitive values per group |
| T-closeness | ≤0.2 | Distribution similarity |
| Re-id Risk | ≤10% | Probability of identification |

#### Summarization Metrics
| Metric | Scale | Description |
|--------|-------|-------------|
| Faithfulness | 0-1 | No hallucinations |
| Coverage | 0-1 | Key info retained |
| ROUGE-1 | 0-1 | Unigram overlap |
| LLM Overall | 1-5 | Overall quality |

---

## 9. Email Attachments Handling

### Where Attachments Are Stored

```
data/
├── bronze/
│   ├── emails/               # Email JSON metadata
│   ├── documents/            # Parsed document JSON
│   └── attachments/          ← ATTACHMENTS GO HERE
│       ├── a1b2c3d4e5f6/     # Unique attachment ID folder
│       │   └── report.pdf
│       ├── f7g8h9i0j1k2/
│       │   └── data.xlsx
│       └── ...
```

### Attachment Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `extract_attachments` | `True` | Enable attachment extraction |
| `attachment_output_dir` | `{bronze_path}/attachments` | Where to save files |
| `max_attachment_size_mb` | 50 MB | Skip files larger than this |

### Supported Attachment Types

By default, these file types are extracted from emails:

```
.pdf, .docx, .doc, .xlsx, .xls, .pptx, .ppt, .txt, .csv, .rtf
```

### Pipeline Flow for Attachments

```
PST File
    │
    ▼
┌─────────────────────────────────────────┐
│  1. Email Extraction                    │
│  ├── Body text → bronze/emails/         │
│  └── Attachments → bronze/attachments/  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. Document Parser                     │
│  ├── PDF → Text extraction              │
│  ├── DOCX → Text extraction             │
│  └── All → bronze/documents/ (JSON)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. Silver Layer Processing             │
│  ├── Chunk all text                     │
│  ├── Detect PII                         │
│  ├── Anonymize                          │
│  └── Summarize (if OpenAI mode)         │
└─────────────────────────────────────────┘
```

### Example: Email with Attachment

After processing an email with a PDF attachment:

**1. Email JSON** (`bronze/emails/email_001.json`):
```json
{
  "message_id": "abc123",
  "subject": "Q4 Report",
  "sender": "john@company.com",
  "sender_email": "john@company.com",
  "recipients_to": ["team@company.com"],
  "sent_time": "2024-01-15T10:30:00",
  "body_text": "Please find attached the Q4 report...",
  "has_attachments": true,
  "attachment_count": 1
}
```

**2. Attachment file** (`bronze/attachments/a1b2c3/Q4_Report.pdf`):
- Original PDF file saved here
- Organized by unique attachment ID

**3. Parsed attachment** (`bronze/documents/Q4_Report.json`):
```json
{
  "doc_id": "doc_Q4_Report",
  "source_file": "bronze/attachments/a1b2c3/Q4_Report.pdf",
  "file_type": "pdf",
  "text": "Quarterly Financial Report\n\nPrepared by John Smith...",
  "pages": 5,
  "extraction_time": "2024-01-20T14:30:00"
}
```

**4. Anonymized chunks** (`silver/chunks/chunk_001.json`):
```json
{
  "chunk_id": "chunk_001",
  "doc_id": "doc_Q4_Report",
  "text_original": "Quarterly Financial Report\n\nPrepared by John Smith...",
  "text_anonymized": "Quarterly Financial Report\n\nPrepared by [PERSON_1]...",
  "summary": "Q4 financial report covering revenue and expenses...",
  "pii_entities": [
    {"text": "John Smith", "type": "PERSON", "confidence": 0.95}
  ],
  "pii_count": 1
}
```

### Linking Emails to Attachments

The relationship is tracked via:
- `email.has_attachments` → Boolean flag
- `email.attachment_count` → Number of attachments
- `attachment.attachment_id` → Unique ID linking to folder name
- `document.source_file` → Path to original attachment

---

## 10. Rerun Behavior & Incremental Processing

### Current Behavior

The pipeline uses **deterministic filenames** based on document/message IDs:

| Layer | File Naming | On Rerun |
|-------|-------------|----------|
| **Bronze** | `{message_id}.json` for emails | Overwritten |
| **Bronze** | `{doc_id}.json` for documents | Overwritten |
| **Silver** | `{doc_id}_chunk_{index}.json` | Overwritten |

**Key Points:**
- **No Duplicates**: Running the pipeline multiple times won't create duplicate files
- **Idempotent**: Same input always produces the same output files
- **Full Reprocessing**: Currently, all documents are reprocessed on every run
- **Metadata Appended**: Processing logs track each run separately

### File Naming Examples

```
# Email from PST
bronze/emails/2024/01/MSG-12345-abc.json    # Based on message_id

# Document
bronze/documents/pdf/DOC-report-xyz.json     # Based on doc_id (filename hash)

# Chunks in Silver
silver/chunks/DOC-report-xyz_chunk_0.json
silver/chunks/DOC-report-xyz_chunk_1.json
```

### Processing Logs

Each run appends to metadata files (not overwrites):

```
data/bronze/metadata/ingestion_log.json       # Bronze processing history
data/silver/metadata/processing_stats.json    # Silver processing history
```

Example log entry:
```json
{
  "start_time": "2024-01-15T10:30:00",
  "end_time": "2024-01-15T10:35:00",
  "documents_processed": 50,
  "chunks_created": 245,
  "pii_detected": 89,
  "errors": 0
}
```

### When to Rerun

| Scenario | Action |
|----------|--------|
| Added new documents | Rerun - only new files get new entries |
| Changed configuration | Rerun - existing files get updated |
| Fixed source documents | Rerun - updated content overwrites old |
| Same data, different day | Safe to rerun - produces same results |

### Best Practices

1. **Development**: Rerun freely - files are overwritten, not duplicated
2. **Production**: Keep metadata logs to track processing history
3. **Large Datasets**: Consider clearing Bronze/Silver before major reruns to remove orphaned files
4. **Incremental**: Add new documents to `input/` and rerun - existing Bronze files are updated, new ones added

### Clearing Data for Fresh Run

```bash
# Clear all processed data (keep input)
rm -rf data/bronze data/silver data/gold

# Or clear specific layers
rm -rf data/silver   # Reprocess from Bronze
rm -rf data/bronze   # Re-extract from source
```

---

## 11. Troubleshooting

### Common Issues

#### "No module named 'presidio_analyzer'"
```bash
pip install presidio-analyzer presidio-anonymizer
```

#### "Can't find model 'en_core_web_lg'"
```bash
python -m spacy download en_core_web_lg
python -m spacy download nl_core_news_lg
```

#### "OpenAI API key not configured"
```bash
export OPENAI_API_KEY='sk-your-key-here'
# Or add to .env file
```

#### "PIPELINE_MODE not recognized"
Valid modes are: `openai`, `local`, `hybrid`
```bash
export PIPELINE_MODE=openai
```

#### Python version mismatch
```bash
# Check version
python --version

# Use specific version
python3.12 -m venv .venv
```

#### Memory issues with large files
```bash
# Reduce chunk size
python src/pipeline/run_ingestion.py \
    --documents data/input/documents \
    --chunk-size 256
```

### Getting Help

```bash
# Pipeline help
python src/pipeline/run_ingestion.py --help

# Evaluation help
python src/pipeline/evaluate_quality.py --help

# Privacy metrics help
python src/pipeline/evaluate_privacy.py --help
```

---

## Quick Reference Commands

```bash
# Setup
source .venv/bin/activate
export OPENAI_API_KEY='your-key'
export PIPELINE_MODE='openai'

# Run pipeline
python src/pipeline/run_ingestion.py --documents data/input/documents --output data

# Evaluate
python src/pipeline/evaluate_quality.py --demo
python src/pipeline/evaluate_quality.py --all --silver data/silver

# Privacy check
python src/pipeline/evaluate_privacy.py --silver data/silver
```

---

## Next Steps

After running the pipeline:

1. **Review Silver Layer**: Check `data/silver/chunks/` for anonymized content
2. **Evaluate Quality**: Run evaluation scripts to measure accuracy
3. **Adjust Configuration**: Tune thresholds based on results
4. **Build Knowledge Graph**: Use Gold layer for RAG/GraphRAG (future phase)

For architecture details, see: `docs/phase_guides/WithPathRagAndReact.md`
