# Pipeline Architecture: Structuring Unstructured Expert Knowledge

## Overview

This document describes the complete architecture for extracting, processing, and querying knowledge from email archives (PST files) using a combination of PathRAG and GraphRAG approaches.

## Pipeline Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BRONZE LAYER                                    │
│                         (Raw Data Extraction)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Extract emails from PST files                                             │
│  • Extract and store attachments with metadata                               │
│  • Establish email-attachment relationships                                  │
│  • Store raw data in structured JSON format                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SILVER LAYER                                    │
│                    (Processing & Entity Extraction)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Smart routing for attachment processing (local vs OpenAI Vision)          │
│  • Content extraction from all document types                                │
│  • Thread-aware chunking (preserves conversation context)                    │
│  • Entity extraction (spaCy NER + LLM-based)                                │
│  • Relationship extraction (co-occurrence + semantic)                        │
│  • PII detection and anonymization                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               GOLD LAYER                                     │
│              (Graph Construction, Summarization, Indexing)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Knowledge Graph construction (entities + relationships)                   │
│  • GraphRAG: Community detection (Leiden algorithm)                          │
│  • GraphRAG: Hierarchical community summarization                            │
│  • PathRAG: Path indexing and pre-computation                               │
│  • Vector embeddings for all content                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL LAYER                                    │
│                    (Query Processing & Answer Generation)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • PathRAG: Path-based reasoning retrieval                                   │
│  • GraphRAG: Community-based context retrieval                               │
│  • Hybrid retrieval combining multiple strategies                            │
│  • LLM-based answer generation with citations                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Bronze Layer - Raw Data Extraction

### Purpose
Extract and store raw data from PST email archives, preserving all metadata and relationships.

### Directory Structure

```
data/bronze/
├── emails/
│   └── {year}/{month}/{message_id}.json
├── attachments/
│   └── {attachment_id}/
│       ├── original/
│       │   └── {filename}
│       └── metadata.json
└── attachment_index.json
```

### Email Schema

```json
{
  "message_id": "abc123def456",
  "source_pst": "data/input/pst/archive.pst",
  "folder_path": "Inbox/Projects",
  "subject": "RE: Project Update",
  "sender": "John Doe",
  "sender_email": "john.doe@company.com",
  "recipients_to": ["jane@company.com"],
  "recipients_cc": ["team@company.com"],
  "sent_time": "2013-11-06T15:31:28.838000",
  "received_time": "2013-11-06T15:31:00",
  "body_text": "...",
  "conversation_id": "Project Update",
  "has_attachments": true,
  "attachment_count": 2,
  "importance": "normal",
  "language": "en",
  "extraction_time": "2024-01-15T10:30:00",
  "attachments": [
    {
      "attachment_id": "att_001",
      "filename": "report.pdf",
      "content_type": "application/pdf",
      "size": 245000
    }
  ]
}
```

### Attachment Metadata Schema

```json
{
  "attachment_id": "att_001",
  "source_email_id": "abc123def456",
  "source_email_subject": "RE: Project Update",
  "source_email_date": "2013-11-06",
  "source_email_sender": "john.doe@company.com",
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "size": 245000,
  "file_hash": "sha256:a1b2c3d4...",
  "extraction_time": "2024-01-15T10:30:00",
  "analysis": {
    "is_image_based": false,
    "has_extractable_text": true,
    "estimated_pages": 5,
    "chars_per_page": 2500.0,
    "complexity_score": 0.3,
    "recommended_processor": "local_parser",
    "routing_reason": "Text-based PDF (2500 chars/page)"
  }
}
```

### Attachment Index Schema

```json
{
  "by_email": {
    "abc123def456": ["att_001", "att_002"],
    "def456ghi789": ["att_003"]
  },
  "by_type": {
    "application/pdf": ["att_001", "att_003"],
    "image/png": ["att_002"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ["att_004"]
  },
  "by_processor": {
    "local_parser": ["att_001", "att_004"],
    "openai_vision": ["att_002", "att_003"]
  },
  "statistics": {
    "total_attachments": 100,
    "total_size_bytes": 52428800,
    "by_type_count": {
      "pdf": 45,
      "xlsx": 30,
      "png": 15,
      "docx": 10
    }
  }
}
```

---

## 2. Silver Layer - Processing & Entity Extraction

### Purpose
Process raw data, extract content from attachments, chunk text, and extract entities/relationships.

### Directory Structure

```
data/silver/
├── attachment_content/
│   └── {attachment_id}.json
├── thread_chunks/
│   └── conv:{thread_subject}_{chunk_index}.json
├── email_chunks/
│   └── {message_id}_{chunk_index}.json
├── email_summaries/
│   └── {email_id}.json
├── thread_summaries/
│   └── {thread_id}.json
├── pii_mappings/
│   └── {message_id}.json
└── metadata/
    └── processing_stats.json
```

### Attachment Content Schema

```json
{
  "attachment_id": "att_001",
  "source_email_id": "abc123def456",
  "filename": "report.pdf",
  "extraction_method": "local_parser",
  "extraction_model": null,
  "extracted_text": "Full extracted text content...",
  "text_length": 12500,
  "pages_processed": 5,
  "extraction_time": "2024-01-15T10:35:00",
  "extraction_cost": 0.0,
  "extraction_success": true,
  "error_message": null
}
```

### Chunk Schema

```json
{
  "chunk_id": "abc123def456_0",
  "thread_id": "conv:Project Update",
  "chunk_index": 0,
  "text_original": "Original text with PII...",
  "text_anonymized": "Text with [PERSON_1] and [EMAIL_1]...",
  "token_count": 485,
  "thread_subject": "project update",
  "thread_participants": ["John Doe", "Jane Smith"],
  "thread_email_count": 5,
  "email_position": "3/5",
  "pii_entities": [
    {
      "text": "John Doe",
      "type": "PERSON",
      "start": 45,
      "end": 53,
      "confidence": 0.95,
      "method": "presidio"
    }
  ],
  "pii_count": 8,
  "kg_entities": [
    {
      "text": "WFG Project",
      "type": "PROJECT",
      "pathrag_type": "concept",
      "start": 120,
      "end": 131,
      "confidence": 0.85,
      "source": "spacy",
      "is_pii": false
    }
  ],
  "kg_relationships": [
    {
      "source": "John Doe",
      "source_type": "PERSON",
      "target": "WFG Project",
      "target_type": "PROJECT",
      "relationship": "WORKS_ON",
      "confidence": 0.75,
      "evidence": "John mentioned working on WFG"
    }
  ],
  "has_attachments": true,
  "attachment_count": 1,
  "attachment_filenames": ["report.pdf"],
  "language": "en",
  "processing_time": "2024-01-15T10:40:00"
}
```

### Attachment Routing Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FILE ROUTING DECISION TREE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALWAYS LOCAL PARSER:                                                        │
│  • .txt, .csv, .json, .xml, .html                                           │
│  • .xlsx, .xls (openpyxl)                                                   │
│  • .docx, .doc (python-docx)                                                │
│  • .pptx, .ppt (python-pptx)                                                │
│                                                                              │
│  ALWAYS OPENAI VISION:                                                       │
│  • .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp, .heic                      │
│                                                                              │
│  NEEDS ANALYSIS (.pdf):                                                      │
│  ├── Try text extraction with pypdf                                         │
│  ├── If chars_per_page > 100 → LOCAL PARSER                                 │
│  ├── If no embedded fonts → OPENAI VISION (scanned)                         │
│  ├── If image_ratio > 70% → OPENAI VISION                                   │
│  ├── If complexity_score > 0.6 → OPENAI VISION                              │
│  └── Otherwise → LOCAL PARSER                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Gold Layer - Graph Construction & Indexing

### Purpose
Build the knowledge graph, detect communities, generate summaries, and create indexes for retrieval.

### Directory Structure

```
data/gold/
├── knowledge_graph/
│   ├── nodes.json
│   ├── edges.json
│   ├── graph.graphml
│   └── graph_stats.json
├── communities/
│   ├── level_0/
│   │   └── community_{id}.json
│   ├── level_1/
│   │   └── community_{id}.json
│   └── level_2/
│       └── community_{id}.json
├── path_index/
│   ├── entity_paths.json
│   ├── chunk_paths.json
│   └── path_statistics.json
├── embeddings/
│   ├── chunk_embeddings.npy
│   ├── chunk_ids.json
│   ├── entity_embeddings.npy
│   ├── entity_ids.json
│   └── embedding_config.json
├── attachment_summaries/
│   └── {attachment_id}.json
└── vector_store/
    └── chroma_db/
```

### Knowledge Graph Node Types

| Node Type | Description | Properties |
|-----------|-------------|------------|
| `EMAIL` | Email message | message_id, subject, date, sender |
| `PERSON` | Person entity | name, email, role |
| `ORGANIZATION` | Company/Org | name, type |
| `PROJECT` | Project/Initiative | name, status |
| `CONCEPT` | Domain concept | name, category |
| `DOCUMENT` | Attachment | filename, type, size |
| `CHUNK` | Text chunk | chunk_id, text, embedding |
| `THREAD` | Email thread | thread_id, subject, email_count |

### Knowledge Graph Edge Types

| Edge Type | Source → Target | Description |
|-----------|-----------------|-------------|
| `SENT_BY` | EMAIL → PERSON | Email sender |
| `SENT_TO` | EMAIL → PERSON | Email recipient |
| `CC_TO` | EMAIL → PERSON | CC recipient |
| `HAS_ATTACHMENT` | EMAIL → DOCUMENT | Email has attachment |
| `PART_OF_THREAD` | EMAIL → THREAD | Email belongs to thread |
| `MENTIONS` | CHUNK → ENTITY | Chunk mentions entity |
| `WORKS_ON` | PERSON → PROJECT | Person works on project |
| `WORKS_AT` | PERSON → ORGANIZATION | Employment |
| `DISCUSSES` | EMAIL → CONCEPT | Email discusses topic |
| `CONTAINS` | DOCUMENT → CONCEPT | Document contains info |
| `RELATED_TO` | ENTITY → ENTITY | Generic relationship |
| `CO_OCCURS` | ENTITY → ENTITY | Entities co-occur in text |

### Community Schema (GraphRAG)

```json
{
  "community_id": "comm_l0_001",
  "level": 0,
  "parent_community": "comm_l1_005",
  "child_communities": [],
  "entities": [
    {"id": "ent_001", "name": "WFG Project", "type": "PROJECT"},
    {"id": "ent_002", "name": "HUD Validation", "type": "PROCESS"}
  ],
  "entity_count": 15,
  "edge_count": 45,
  "density": 0.43,
  "summary": "This community focuses on the WFG (Wells Fargo) project, specifically around HUD document validation. Key activities include OCR processing of loan documents, validation accuracy tracking, and mapping loan numbers to borrower records. Main participants are Muhammad Rafiq and the document processing team.",
  "key_topics": ["HUD validation", "OCR processing", "loan documents"],
  "summary_embedding": [0.123, -0.456, ...],
  "generated_at": "2024-01-15T11:00:00"
}
```

### Path Index Schema (PathRAG)

```json
{
  "paths": [
    {
      "path_id": "path_001",
      "source_entity": {"id": "ent_001", "name": "Muhammad Rafiq", "type": "PERSON"},
      "target_entity": {"id": "ent_015", "name": "OCR Accuracy", "type": "METRIC"},
      "path": [
        {"node": "Muhammad Rafiq", "type": "PERSON"},
        {"edge": "SENT", "type": "relationship"},
        {"node": "Email_abc123", "type": "EMAIL"},
        {"edge": "HAS_ATTACHMENT", "type": "relationship"},
        {"node": "MappedData.xlsx", "type": "DOCUMENT"},
        {"edge": "CONTAINS", "type": "relationship"},
        {"node": "OCR Accuracy", "type": "METRIC"}
      ],
      "path_length": 3,
      "evidence_chunks": ["chunk_001", "chunk_020"],
      "path_weight": 0.85
    }
  ],
  "statistics": {
    "total_paths": 5000,
    "avg_path_length": 2.8,
    "max_path_length": 5,
    "paths_by_type": {
      "PERSON→PROJECT": 1200,
      "PERSON→DOCUMENT": 800,
      "PROJECT→METRIC": 500
    }
  }
}
```

---

## 4. Retrieval Layer - Query Processing

### Purpose
Process user queries using PathRAG and GraphRAG retrieval strategies, and generate answers with LLM.

### Retrieval Strategies

#### 4.1 PathRAG Retrieval

```
Query: "What were the issues with WFG HUD validation?"

Step 1: Extract entities from query
        → [WFG, HUD, validation]

Step 2: Find entities in graph
        → WFG Project (ent_001), HUD Validation (ent_015)

Step 3: Find paths between query entities
        → Path: WFG → INVOLVES → HUD Validation → HAS_METRIC → 88% accuracy

Step 4: Traverse paths and collect evidence chunks
        → [chunk_001, chunk_020, chunk_035]

Step 5: Return chunks with path context
```

#### 4.2 GraphRAG Retrieval

```
Query: "What were the issues with WFG HUD validation?"

Step 1: Embed query
        → query_embedding = embed("What were the issues...")

Step 2: Find relevant communities by summary similarity
        → Community L0_001 (WFG/HUD), L0_003 (OCR Processing)

Step 3: Get community summaries for context
        → "This community focuses on WFG HUD validation..."

Step 4: Map to source chunks via community entities
        → [chunk_001, chunk_020, chunk_035]

Step 5: Return summaries + chunks
```

#### 4.3 Hybrid Retrieval

```
Query: "What were the issues with WFG HUD validation?"

Step 1: Run PathRAG retrieval → chunks_path
Step 2: Run GraphRAG retrieval → chunks_graph
Step 3: Run Vector similarity search → chunks_vector
Step 4: Merge and deduplicate
Step 5: Re-rank by relevance
Step 6: Return top-K chunks with context
```

### Answer Generation

```
System Prompt:
You are an expert assistant analyzing email archives. Use the provided context
to answer questions accurately. Always cite your sources.

Context:
[PathRAG paths]
[GraphRAG community summaries]
[Relevant text chunks]

Question: {user_query}

Answer: {LLM generates answer with citations}
```

---

## 5. Implementation Modules

### Module Overview

| Layer | Module | File | Status |
|-------|--------|------|--------|
| Bronze | PST Extractor | `src/ingestion/pst_extractor.py` | ✅ Done |
| Bronze | Attachment Storage | `src/ingestion/attachment_storage.py` | ✅ Done |
| Silver | Attachment Router | `src/ingestion/attachment_router.py` | ✅ Done |
| Silver | OpenAI Vision Extractor | `src/ingestion/openai_vision_extractor.py` | ✅ Done |
| Silver | Attachment Processor | `src/ingestion/attachment_processor.py` | ✅ Done |
| Silver | Thread-Aware Processor | `src/anonymization/thread_aware_processor.py` | ✅ Done |
| Gold | Graph Builder | `src/graph/graph_builder.py` | ✅ Done |
| Gold | Community Detector | `src/graph/community_detector.py` | ✅ Done |
| Gold | Path Indexer | `src/graph/path_indexer.py` | ✅ Done |
| Gold | Embedding Generator | `src/graph/embedding_generator.py` | ✅ Done |
| Gold | Gold Pipeline Runner | `src/pipeline/run_gold_indexing.py` | ✅ Done |
| Retrieval | Retrieval Tools | `src/retrieval/retrieval_tools.py` | ✅ Done |
| Retrieval | ReAct Retriever | `src/retrieval/react_retriever.py` | ✅ Done |
| Retrieval | Hybrid Retriever | `src/retrieval/hybrid_retriever.py` | ✅ Done |

### Dependencies

```
# Core
python >= 3.10
pypff  # PST extraction

# Document Processing
pypdf  # PDF text extraction
python-docx  # DOCX extraction
openpyxl  # XLSX extraction
python-pptx  # PPTX extraction

# NLP & ML
spacy  # NER
presidio-analyzer  # PII detection
tiktoken  # Token counting
sentence-transformers  # Embeddings (alternative)

# Graph
networkx  # Graph operations
python-igraph  # Leiden algorithm
leidenalg  # Community detection

# Vector Store
chromadb  # Vector storage
faiss-cpu  # Alternative vector store

# LLM
openai  # OpenAI API
azure-identity  # Azure OpenAI

# Utilities
pydantic  # Data validation
tqdm  # Progress bars
```

---

## 6. Cost Estimation

### OpenAI API Costs (per 100 emails)

| Operation | Model | Est. Tokens | Cost |
|-----------|-------|-------------|------|
| Vision OCR (10 scanned PDFs) | GPT-4o | ~50K input | $0.50-1.50 |
| Entity Extraction (LLM) | GPT-4o-mini | ~100K | $0.10-0.20 |
| Community Summaries | GPT-4o-mini | ~50K | $0.05-0.10 |
| Embeddings | text-embedding-3-small | ~200K | $0.02-0.04 |
| Answer Generation | GPT-4o | ~10K/query | $0.05-0.10/query |

**Total Processing Cost: ~$0.70-2.00 per 100 emails**

### Storage Estimates

| Data Type | Size per 100 emails |
|-----------|---------------------|
| Bronze (raw) | ~50-100 MB |
| Silver (processed) | ~20-50 MB |
| Gold (graph + embeddings) | ~100-200 MB |
| Vector Store | ~50-100 MB |

---

## 7. Configuration

### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Or OpenAI Direct
OPENAI_API_KEY=your_key

# Processing
MAX_CHUNK_TOKENS=512
OVERLAP_TOKENS=50
MIN_CHARS_PER_PAGE_FOR_OCR=100
COMPLEXITY_THRESHOLD=0.6

# Graph
COMMUNITY_RESOLUTION=1.0
MAX_PATH_LENGTH=5
MIN_PATH_WEIGHT=0.5
```

### Pipeline Configuration (config.yaml)

```yaml
bronze:
  input_path: data/input/pst/
  output_path: data/bronze/

silver:
  input_path: data/bronze/
  output_path: data/silver/
  chunking:
    max_tokens: 512
    overlap: 50
  attachment_routing:
    min_chars_per_page: 100
    complexity_threshold: 0.6
    use_openai_vision: true
  entity_extraction:
    strategy: spacy  # spacy, llm, hybrid

gold:
  input_path: data/silver/
  output_path: data/gold/
  graph:
    store: networkx  # networkx, neo4j
  communities:
    algorithm: leiden
    resolution: 1.0
    levels: 3
  paths:
    max_length: 5
    min_weight: 0.5
  embeddings:
    model: text-embedding-3-small

retrieval:
  strategies:
    - pathrag
    - graphrag
    - vector
  top_k: 10
  rerank: true
```

---

## 8. Running the Pipeline

### Full Pipeline

```bash
# Step 1: Bronze - Extract from PST
python -m src.pipeline.run_bronze_extraction \
  --input data/input/pst/ \
  --output data/bronze/

# Step 2: Silver - Process and extract entities
python -m src.pipeline.run_silver_processing \
  --bronze data/bronze/ \
  --silver data/silver/ \
  --process-attachments \
  --kg-strategy hybrid

# Step 3: Gold - Build graph and indexes
python -m src.pipeline.run_gold_indexing \
  --silver data/silver/ \
  --gold data/gold/ \
  --build-communities \
  --build-paths \
  --generate-embeddings

# Step 4: Query
python -m src.pipeline.run_query \
  --gold data/gold/ \
  --strategy hybrid \
  --query "What were the issues with WFG validation?"
```

### Development/Testing

```bash
# Test with limited emails
python -m src.pipeline.run_thread_processing \
  --bronze data/bronze_test/ \
  --silver data/silver_test/ \
  --process-attachments \
  --max-emails 100
```

---

## 9. Future Enhancements

1. **Real-time Ingestion**: Support for live email monitoring
2. **Multi-language Support**: NER and processing for non-English emails
3. **Advanced OCR**: Integration with specialized OCR services
4. **Graph Database**: Migration to Neo4j for production scale
5. **Incremental Updates**: Delta processing for new emails
6. **User Feedback Loop**: Learning from user corrections
7. **Visualization**: Interactive graph exploration UI
