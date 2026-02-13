# Data Directory Structure

Place your source files in the `input/` folder and the pipeline will process them through the medallion architecture layers.

## Directory Structure

```
data/
├── input/                    # PUT YOUR SOURCE FILES HERE
│   ├── pst/                  # PST files (Outlook email archives)
│   │   └── emails.pst
│   └── documents/            # Documents (PDF, DOCX, XLSX, TXT, etc.)
│       ├── report.pdf
│       ├── notes.docx
│       └── data.xlsx
│
├── bronze/                   # Raw extracted data (auto-generated)
│   ├── emails/               # Extracted emails as JSON
│   ├── documents/            # Parsed documents as JSON
│   └── attachments/          # Email attachments
│
├── silver/                   # Processed & anonymized data (auto-generated)
│   ├── chunks/               # Text chunks with PII removed
│   └── metadata/             # Processing metadata
│
├── gold/                     # Business-ready data (auto-generated)
│   ├── embeddings/           # Vector embeddings
│   └── knowledge_graph/      # Entity relationships
│
└── evaluation/               # Evaluation files
    └── ground_truth_sample.json
```

## Supported File Types

### PST Files
- Microsoft Outlook PST archives (.pst)
- Contains emails, contacts, calendar items

### Documents
| Extension | Description |
|-----------|-------------|
| `.pdf`    | PDF documents |
| `.docx`   | Microsoft Word |
| `.doc`    | Legacy Word |
| `.xlsx`   | Microsoft Excel |
| `.xls`    | Legacy Excel |
| `.pptx`   | PowerPoint |
| `.txt`    | Plain text |
| `.rtf`    | Rich text |
| `.html`   | HTML files |
| `.eml`    | Email files |
| `.msg`    | Outlook messages |

## Running the Pipeline

### 1. Process PST Files
```bash
source .venv/bin/activate

# Process PST to Bronze layer
python src/pipeline/run_ingestion.py \
    --pst data/input/pst/emails.pst \
    --output data
```

### 2. Process Documents
```bash
# Process documents to Bronze layer
python src/pipeline/run_ingestion.py \
    --documents data/input/documents \
    --output data
```

### 3. Full Pipeline (Bronze → Silver)
```bash
# Run full pipeline with anonymization
python src/pipeline/run_ingestion.py \
    --pst data/input/pst/emails.pst \
    --documents data/input/documents \
    --output data \
    --process-silver
```

### 4. Using OpenAI Mode
```bash
# Set API key first
export OPENAI_API_KEY='your-api-key'
export PIPELINE_MODE='openai'

# Then run pipeline
python src/pipeline/run_ingestion.py \
    --documents data/input/documents \
    --output data \
    --process-silver
```

## Evaluate Results

### PII Detection Accuracy
```bash
python src/pipeline/evaluate_anonymization.py \
    --ground-truth data/evaluation/ground_truth_sample.json
```

### Privacy Metrics
```bash
python src/pipeline/evaluate_privacy.py \
    --silver data/silver
```

## Configuration

Set processing mode via environment variable:
```bash
export PIPELINE_MODE=openai   # Use OpenAI API (default)
export PIPELINE_MODE=local    # Use local models only
export PIPELINE_MODE=hybrid   # Combine local + OpenAI
```

Or copy `.env.template` to `.env` and configure there.
