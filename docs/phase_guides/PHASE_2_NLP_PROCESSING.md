# Phase 2: NLP Processing

**Duration:** Weeks 3-4
**Goal:** Transform raw text into clean, enriched chunks ready for search

---

## Overview

### What We're Building

In this phase, we process the raw text from Phase 1:
1. **Semantic Chunking**: Split documents into meaningful pieces
2. **Named Entity Recognition (NER)**: Find people, organizations, locations
3. **PII Anonymization**: Protect sensitive personal information
4. **Summarization**: Create concise summaries

### Why This Matters

```
RAW DOCUMENT (10,000 words)
         │
         ▼
    ┌─────────┐
    │ CHUNKING │ → Pieces that fit in LLM context
    └─────────┘
         │
         ▼
    ┌─────────┐
    │   NER   │ → Know WHO and WHAT is mentioned
    └─────────┘
         │
         ▼
    ┌─────────┐
    │   PII   │ → Safe to process (no personal data exposed)
    └─────────┘
         │
         ▼
    ┌─────────┐
    │ SUMMARY │ → Quick understanding without reading all
    └─────────┘
         │
         ▼
PROCESSED CHUNKS (ready for RAG)
```

---

## Prerequisites

### From Phase 1

- [ ] Emails in `/mnt/datalake/bronze/emails/`
- [ ] Documents in `/mnt/datalake/bronze/documents/`
- [ ] Language tags on all content

### Additional Dependencies

```python
# NLP libraries (already in requirements.txt)
spacy>=3.5.0
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
langchain>=0.1.0
```

### Models to Download

```python
# English NER model (large, transformer-based)
python -m spacy download en_core_web_trf

# Dutch NER model
python -m spacy download nl_core_news_lg
```

---

## Step 1: Semantic Chunking

### What We're Doing
Breaking long documents into smaller pieces that preserve meaning.

### Why Semantic Chunking (Not Fixed-Size)

**Bad: Fixed-size chunking**
```
Chunk 1: "...the meeting was held on Tuesday. John Smith, the CEO, announced that"
Chunk 2: "the company would be acquiring TechCorp for $500 million. This decision..."
```
Sentence is split! Context is lost!

**Good: Semantic chunking**
```
Chunk 1: "...the meeting was held on Tuesday."
Chunk 2: "John Smith, the CEO, announced that the company would be acquiring
          TechCorp for $500 million. This decision..."
```
Complete thoughts stay together!

### How Semantic Chunking Works

```python
# 1. Split into sentences
sentences = split_into_sentences(document)

# 2. Create embeddings for each sentence
embeddings = embed(sentences)

# 3. Find semantic breakpoints
# Where consecutive sentences are very different → break
breakpoints = find_breakpoints(embeddings, threshold=0.5)

# 4. Group sentences between breakpoints into chunks
chunks = group_by_breakpoints(sentences, breakpoints)
```

### Instructions

1. **Run the Semantic Chunking Notebook**

   ```
   notebooks/02_nlp_processing/01_semantic_chunking.py
   ```

2. **Key Configuration**

   ```python
   from processors.chunking import SemanticChunker, ChunkingConfig

   config = ChunkingConfig(
       min_chunk_size=256,        # Minimum tokens per chunk
       max_chunk_size=1024,       # Maximum tokens per chunk
       target_chunk_size=512,     # Ideal chunk size
       overlap_size=50,           # Overlap between chunks (for context)
       similarity_threshold=0.5,  # When to break (lower = more breaks)
   )

   chunker = SemanticChunker(
       azure_endpoint=AZURE_OPENAI_ENDPOINT,
       api_key=AZURE_OPENAI_KEY,
       config=config
   )
   ```

3. **Processing Flow**

   ```python
   # Load documents from Bronze
   docs_df = spark.read.format("delta").load("/mnt/datalake/bronze/documents")

   # Process each document
   all_chunks = []
   for doc in docs_df.collect():
       chunks = chunker.chunk_document(
           text=doc.content,
           metadata={
               "doc_id": doc.doc_id,
               "file_name": doc.file_name,
               "language": doc.language
           }
       )
       all_chunks.extend(chunks)

   # Save to Silver
   chunks_df = spark.createDataFrame(all_chunks)
   chunks_df.write.format("delta").save("/mnt/datalake/silver/chunks")
   ```

### Expected Output

```
Input:  500,000 documents
Output: 2,000,000 chunks

Average chunk size: 450 tokens
Chunk size distribution:
  - 256-400 tokens: 30%
  - 400-600 tokens: 50%
  - 600-1024 tokens: 20%
```

### Chunk Schema

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Unique chunk identifier |
| doc_id | string | Parent document ID |
| text | string | Chunk content |
| chunk_index | int | Position in document (0, 1, 2...) |
| token_count | int | Number of tokens |
| start_char | int | Start position in original doc |
| end_char | int | End position in original doc |

---

## Step 2: Named Entity Recognition (NER)

### What We're Doing
Finding and tagging entities like people, organizations, and locations.

### Why NER Matters

```
Without NER:
"John met with the CEO at Microsoft HQ in Seattle."
(Just text - we don't know who/what/where)

With NER:
"[PERSON: John] met with the [TITLE: CEO] at [ORG: Microsoft] HQ in [LOCATION: Seattle]."
(Now we can search for all mentions of "Microsoft" or "Seattle")
```

### Entity Types We Extract

| Entity Type | Examples | Why It Matters |
|-------------|----------|----------------|
| PERSON | John Smith, Marie Curie | Who is involved |
| ORG | Microsoft, KU Leuven | What companies/institutions |
| LOCATION | Brussels, New York | Where things happen |
| DATE | January 2024, last Tuesday | When things happen |
| MONEY | $500 million, €1.2M | Financial context |
| PRODUCT | iPhone, Azure | What products discussed |

### Instructions

1. **Run the NER Extraction Notebook**

   ```
   notebooks/02_nlp_processing/02_ner_extraction.py
   ```

2. **How We Handle Multilingual NER**

   ```python
   from processors.ner import MultilingualNER

   ner = MultilingualNER()

   # Automatically selects model based on language
   def extract_entities(text, language):
       if language == 'en':
           # Use English transformer model
           return ner.extract_english(text)
       elif language == 'nl':
           # Use Dutch model
           return ner.extract_dutch(text)
       else:
           # Fallback to English
           return ner.extract_english(text)
   ```

3. **Processing Flow**

   ```python
   # Load chunks from Silver
   chunks_df = spark.read.format("delta").load("/mnt/datalake/silver/chunks")

   # Extract entities from each chunk
   def process_chunk(chunk):
       entities = ner.extract(chunk.text, chunk.language)
       return {
           "chunk_id": chunk.chunk_id,
           "entities": entities,
           "entity_count": len(entities)
       }

   # Apply to all chunks
   ner_results = chunks_df.rdd.map(process_chunk).toDF()

   # Save results
   ner_results.write.format("delta").save("/mnt/datalake/silver/ner_results")
   ```

4. **Entity Linking (Optional Enhancement)**

   ```python
   # Link mentions to canonical entities
   # "John", "J. Smith", "Mr. Smith" → Same person

   linker = EntityLinker()
   linked_entities = linker.link(entities)
   ```

### Expected Output

```
Entities extracted: 5,000,000
By type:
  - PERSON: 1,500,000
  - ORG: 1,200,000
  - LOCATION: 800,000
  - DATE: 1,000,000
  - OTHER: 500,000

Unique entities: 50,000
```

### NER Result Schema

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Source chunk |
| entity_text | string | The entity text ("John Smith") |
| entity_type | string | Type (PERSON, ORG, etc.) |
| start_pos | int | Start position in chunk |
| end_pos | int | End position in chunk |
| confidence | float | Model confidence (0-1) |

---

## Step 3: PII Anonymization

### What We're Doing
Finding and replacing personal information to protect privacy.

### Why PII Anonymization

**Legal Requirements:**
- GDPR (Europe): Must protect personal data
- CCPA (California): Consumer privacy rights
- Company Policy: Enterprise data governance

**What Counts as PII:**
- Names
- Email addresses
- Phone numbers
- Social Security / National ID numbers
- Bank accounts (IBAN)
- Credit card numbers
- Addresses
- Passport numbers

### How Anonymization Works

```
BEFORE:
"Contact John Smith at john.smith@company.com or call +1-555-123-4567.
 His SSN is 123-45-6789 and IBAN is BE68 5390 0754 7034."

AFTER:
"Contact [PERSON_1] at [EMAIL_1] or call [PHONE_1].
 His SSN is [SSN_1] and IBAN is [IBAN_1]."
```

The mapping is saved so you can reverse if needed:
```json
{
  "PERSON_1": "John Smith",
  "EMAIL_1": "john.smith@company.com",
  "PHONE_1": "+1-555-123-4567"
}
```

### Instructions

1. **Run the Anonymization Notebook**

   ```
   notebooks/02_nlp_processing/03_anonymization.py
   ```

2. **Understanding the Global PII Recognizer**

   ```python
   from processors.anonymization import GlobalPIIAnonymizer, GlobalPIIConfig

   config = GlobalPIIConfig(
       # What to detect
       detect_names=True,
       detect_emails=True,
       detect_phones=True,
       detect_ssn=True,
       detect_iban=True,
       detect_credit_cards=True,

       # International patterns
       phone_patterns=["US", "EU", "UK", "INTERNATIONAL"],
       id_patterns=["US_SSN", "EU_VAT", "UK_NI", "PASSPORT"],

       # Processing
       anonymization_method="replace",  # or "redact", "hash"
       preserve_format=True,  # Keep structure for debugging
   )

   anonymizer = GlobalPIIAnonymizer(config)
   ```

3. **Processing Flow**

   ```python
   # Load chunks
   chunks_df = spark.read.format("delta").load("/mnt/datalake/silver/chunks")

   # Anonymize each chunk
   def anonymize_chunk(chunk):
       result = anonymizer.anonymize(chunk.text)
       return {
           "chunk_id": chunk.chunk_id,
           "text_anonymized": result.anonymized_text,
           "pii_found": result.pii_entities,
           "pii_count": len(result.pii_entities)
       }

   # Apply and save
   anonymized_df = chunks_df.rdd.map(anonymize_chunk).toDF()
   anonymized_df.write.format("delta").save("/mnt/datalake/silver/chunks_anonymized")
   ```

4. **Reversible vs Irreversible**

   ```python
   # Reversible (for internal use)
   # Stores mapping, can recover original
   anonymizer.anonymize(text, reversible=True)

   # Irreversible (for external sharing)
   # No way to get original back
   anonymizer.anonymize(text, reversible=False)
   ```

### Expected Output

```
PII detected and anonymized:
  - Names: 500,000 instances
  - Emails: 200,000 instances
  - Phones: 150,000 instances
  - IBANs: 50,000 instances
  - SSNs: 10,000 instances

Chunks with PII: 60% (1,200,000 chunks)
Chunks without PII: 40% (800,000 chunks)
```

### Anonymization Schema

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Source chunk |
| text_anonymized | string | Text with PII replaced |
| pii_entities | array | List of PII found |
| pii_mapping_id | string | Reference to decryption mapping |

---

## Step 4: Summarization

### What We're Doing
Creating concise summaries of chunks and documents using GPT-4o.

### Why Summarization

1. **Quick Understanding**: Know what a document is about without reading all
2. **Better Search**: Summaries capture main points for retrieval
3. **Hierarchy**: Document summaries → Section summaries → Chunk summaries

### Types of Summaries

| Type | Length | Purpose |
|------|--------|---------|
| Chunk Summary | 1-2 sentences | What this chunk is about |
| Document Summary | 1 paragraph | What the entire document covers |
| Thread Summary | 2-3 sentences | What an email conversation is about |

### Instructions

1. **Run the Summarization Notebook**

   ```
   notebooks/02_nlp_processing/04_summarization.py
   ```

2. **Configuration**

   ```python
   from processors.summarization import Summarizer, SummaryConfig

   config = SummaryConfig(
       model_deployment="gpt-4o",
       temperature=0.3,  # Low for factual summaries
       chunk_summary_max_tokens=100,
       document_summary_max_tokens=300,
   )

   summarizer = Summarizer(
       azure_endpoint=AZURE_OPENAI_ENDPOINT,
       api_key=AZURE_OPENAI_KEY,
       config=config
   )
   ```

3. **Chunk Summarization**

   ```python
   # For each chunk, generate a brief summary
   def summarize_chunk(chunk):
       prompt = f"""Summarize this text in 1-2 sentences:

   {chunk.text}

   Summary:"""

       summary = summarizer.summarize(prompt)
       return {
           "chunk_id": chunk.chunk_id,
           "summary": summary
       }
   ```

4. **Document Summarization (Map-Reduce)**

   For long documents, we use map-reduce:
   ```
   Document (50 pages)
        │
        ├── Chunk 1 → Summary 1
        ├── Chunk 2 → Summary 2
        ├── ...
        └── Chunk 50 → Summary 50
                │
                ▼
        Combine all chunk summaries
                │
                ▼
        Final document summary
   ```

   ```python
   from processors.summarization import MapReduceSummarizer

   map_reduce = MapReduceSummarizer(summarizer)

   # Summarize entire document
   doc_summary = map_reduce.summarize_document(
       chunks=document_chunks,
       max_final_length=300
   )
   ```

5. **Email Thread Summarization**

   ```python
   from processors.summarization import EmailThreadSummarizer

   thread_summarizer = EmailThreadSummarizer(summarizer)

   # Summarize a conversation
   thread_summary = thread_summarizer.summarize(
       emails=thread_emails,
       include_participants=True,
       include_dates=True
   )

   # Output: "Discussion between John and Mary (Jan 5-8, 2024)
   #          about the Q1 budget proposal. Key decision: approved
   #          $50K for marketing."
   ```

### Expected Output

```
Summaries generated:
  - Chunk summaries: 2,000,000
  - Document summaries: 500,000
  - Thread summaries: 150,000

Average generation time: 0.5 seconds per chunk
Total API cost estimate: ~$500
```

### Summary Schema

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Source chunk |
| summary | string | Generated summary |
| summary_tokens | int | Token count of summary |
| model_used | string | gpt-4o |
| generated_at | timestamp | When generated |

---

## Phase 2 Checklist

Before moving to Phase 3, verify:

- [ ] Semantic chunking complete
  - [ ] All documents chunked
  - [ ] Chunk sizes in target range (256-1024)
  - [ ] Overlap applied for context

- [ ] NER extraction complete
  - [ ] English documents processed with EN model
  - [ ] Dutch documents processed with NL model
  - [ ] Entity types correctly identified

- [ ] PII anonymization complete
  - [ ] All PII types detected
  - [ ] Mappings saved (if reversible)
  - [ ] Anonymized chunks saved

- [ ] Summarization complete
  - [ ] Chunk summaries generated
  - [ ] Document summaries generated
  - [ ] Thread summaries generated

---

## Verification Queries

```python
# Check chunk count and sizes
chunks_df = spark.read.format("delta").load("/mnt/datalake/silver/chunks")
print(f"Total chunks: {chunks_df.count()}")
chunks_df.select(avg("token_count"), min("token_count"), max("token_count")).show()

# Check NER results
ner_df = spark.read.format("delta").load("/mnt/datalake/silver/ner_results")
ner_df.groupBy("entity_type").count().show()

# Check anonymization
anon_df = spark.read.format("delta").load("/mnt/datalake/silver/chunks_anonymized")
print(f"Chunks with PII: {anon_df.filter(col('pii_count') > 0).count()}")

# Check summaries
summary_df = spark.read.format("delta").load("/mnt/datalake/silver/summaries")
print(f"Summaries generated: {summary_df.count()}")
```

---

## Cost Estimation

| Operation | API Calls | Est. Cost |
|-----------|-----------|-----------|
| Semantic Chunking | 500K embeddings | ~$50 |
| NER | Local (free) | $0 |
| PII Detection | Local (free) | $0 |
| Summarization | 2M GPT-4o calls | ~$500 |
| **Total** | | **~$550** |

*Costs based on Azure OpenAI pricing as of 2024*

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Chunks too small | Threshold too high | Lower `similarity_threshold` to 0.3 |
| Chunks too large | Threshold too low | Increase `similarity_threshold` to 0.7 |
| NER missing entities | Wrong language model | Check language detection |
| PII not detected | Missing pattern | Add custom regex to config |
| Summarization slow | Too many API calls | Use batching, increase parallelism |
| Rate limits hit | Too fast | Add exponential backoff |

---

## What's Next

In **Phase 3: Vector Index & Basic RAG**, we will:
1. Generate embeddings for all chunks
2. Create Azure AI Search index with HNSW
3. Build basic RAG question-answering system
4. Measure baseline performance (MRR, NDCG)

---

*Phase 2 Complete! Proceed to [Phase 3: Vector Index](./PHASE_3_VECTOR_INDEX.md)*
