# SimpleRAG Pipeline

End-to-End AI Data Pipeline for Email Knowledge Base with RAG retrieval.

**Architecture**: Bronze → Silver → Gold (Medallion Architecture)

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full pipeline
python -m src.simplerag.run_pipeline --all
```

## Running Each Layer

### 1. Bronze Layer (Raw Ingestion)
```bash
# Place source files first
cp your_emails.pst data/simplerag/source/

# Run Bronze ingestion
python -m src.simplerag.run_pipeline --bronze
```

### 2. Silver Layer (AI Processing)
```bash
# Process all Bronze records (skips already processed)
python -m src.simplerag.run_pipeline --silver

# Check progress
ls data/simplerag/silver/processed/*.json | wc -l
```

### 3. Gold Layer (RAG Indexing)
```bash
# Index all Silver records (skips already indexed)
python -m src.simplerag.run_pipeline --gold

# Check index stats
python -m src.simplerag.ask --stats
```

### Run All Layers at Once
```bash
python -m src.simplerag.run_pipeline --all
```

## Querying the Knowledge Base

### Command Line
```bash
# Single question
python -m src.simplerag.ask "What JIRA issues were created?"

# With verbose sources
python -m src.simplerag.ask -v "What VPN access was requested?"

# Interactive chat mode
python -m src.simplerag.ask --interactive

# Check index stats
python -m src.simplerag.ask --stats
```

### Web Interface (Gradio)
```bash
python -m src.simplerag.app
# Opens at http://localhost:7860
```

### Python API
```python
import asyncio
from src.simplerag.run_pipeline import SimpleRAGPipeline

async def main():
    pipeline = SimpleRAGPipeline()
    result = await pipeline.query("Your question here")
    print(result['answer'])
    print(result['sources'])

asyncio.run(main())
```

## Pipeline Layers

### Bronze Layer (Raw Ingestion)
- **Input**: PST, MSG, PDF, DOCX files in `data/simplerag/source/`
- **Output**: `data/simplerag/bronze/emails/*.json`
- **Features**:
  - No AI processing (raw extraction only)
  - Email threading via Message-ID/In-Reply-To
  - Duplicate detection across PST files
  - Attachment extraction (documents only, no icons)

```bash
# Place files in source folder
cp emails.pst data/simplerag/source/

# Run Bronze ingestion
python -m src.simplerag.run_pipeline --bronze
```

### Silver Layer (AI Processing)
- **Input**: Bronze records
- **Output**: `data/simplerag/silver/processed/*.json`
- **Features**:
  - PII anonymization (names → [PERSON_1], emails → [EMAIL_1])
  - Disclaimer/signature removal
  - AI summarization (2-3 sentences)
  - Text chunking for RAG
  - Full lineage tracking

```bash
# Process Bronze → Silver (skips already processed)
python -m src.simplerag.run_pipeline --silver
```

### Gold Layer (RAG Index)
- **Input**: Silver records
- **Output**: `data/simplerag/gold/index/`, `data/simplerag/gold/embeddings/`
- **Features**:
  - Vector embeddings (Azure OpenAI text-embedding-3-small)
  - Semantic search
  - Grounded answer generation
  - Source citations with lineage

```bash
# Index Silver → Gold (skips already indexed)
python -m src.simplerag.run_pipeline --gold
```

## Directory Structure

```
data/simplerag/
├── source/              # Drop files here for ingestion
├── inprogress/          # Files being processed
├── processed/           # Completed source files
├── error/               # Failed files
├── bronze/
│   ├── emails/          # One JSON per email
│   ├── attachments/     # Extracted documents
│   └── metadata/        # Duplicates, message index
├── silver/
│   ├── processed/       # Anonymized, summarized records
│   └── ocr/             # OCR results (if any)
└── gold/
    ├── index/           # Chunk index
    ├── embeddings/      # Vector embeddings (.npz)
    └── metadata/        # Query logs, lineage
```

## Data Templates

Example record structures are documented in:
- `bronze/bronze_email_data_template.json`
- `silver/silver_email_data_template.json`
- `gold/gold_index_template.json`

## Configuration

Environment variables (`.env` file):
```bash
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

## Resume Processing

Both Silver and Gold layers support resuming:
- **Silver**: Skips Bronze records that already have Silver output
- **Gold**: Skips chunks that are already indexed

Simply re-run the command to continue from where you left off:
```bash
python -m src.simplerag.run_pipeline --silver  # Continues processing
python -m src.simplerag.run_pipeline --gold    # Adds new chunks only
```

## Query Modes

| Mode | Description | Usage |
|------|-------------|-------|
| SimpleRAG | Vector similarity search | Default |
| PathRAG | Filter by conversation thread | `thread_id="xxx"` |
| GraphRAG | Filter by sender domain | `sender_domain="company.com"` |

## Monitoring Progress

```bash
# Silver progress
ls data/simplerag/silver/processed/*.json | wc -l

# Gold stats
python -m src.simplerag.ask --stats
```

## Lineage Tracking

Every answer traces back to source:
```
Query Response
  └── Chunk (gold/index)
       └── Silver Record (silver/processed)
            └── Bronze Record (bronze/emails)
                 └── Source File (PST/MSG)
```

---
**SimpleRAG** | Master's Thesis @ KU Leuven | Structuring Unstructured Expert Knowledge
