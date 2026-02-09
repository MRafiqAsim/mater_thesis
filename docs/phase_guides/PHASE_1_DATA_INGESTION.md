# Phase 1: Data Ingestion

**Duration:** Weeks 1-2
**Goal:** Load all enterprise data into the system

---

## Overview

### What We're Building

In this phase, we create the data foundation by:
1. Setting up Azure infrastructure
2. Loading 35 years of emails from PST files
3. Loading documents (PDF, DOCX, XLSX, PPTX)
4. Detecting language (English vs Dutch)

### Why This Matters

- **Single Source of Truth**: All data in one place (Delta Lake)
- **Structured Storage**: Bronze layer preserves raw data
- **Metadata Tracking**: Know where every piece of data came from
- **Language Awareness**: Different NLP models for EN vs NL

---

## Prerequisites

### Azure Resources Needed

| Resource | Purpose | SKU Recommendation |
|----------|---------|-------------------|
| Azure Databricks | Notebook execution | Premium (for Unity Catalog) |
| ADLS Gen2 | Delta Lake storage | Standard LRS |
| Azure OpenAI | LLM & embeddings | S0 |
| Azure AI Search | Vector search | Standard S1 |
| Key Vault | Secret management | Standard |

### Data You Need

- PST files (Outlook email archives)
- PDF documents
- Word documents (.docx)
- Excel files (.xlsx)
- PowerPoint files (.pptx)
- MSG files (individual emails)

### Local Setup

```bash
# Clone the repository
git clone <your-repo>
cd mater_thesis

# Install dependencies locally (for testing)
pip install -r requirements.txt
```

---

## Step 1: Azure Environment Setup

### What We're Doing
Setting up Databricks secrets and storage mounts so notebooks can access Azure services securely.

### Why
- **Security**: API keys shouldn't be in code
- **Portability**: Same code works across environments
- **Compliance**: Enterprise security requirements

### Instructions

1. **Create Databricks Secret Scopes**

   In Databricks workspace, go to `https://<workspace-url>#secrets/createScope`

   Create these scopes:
   ```
   Scope: azure-openai
   Scope: azure-search
   Scope: azure-cosmos
   Scope: azure-storage
   ```

2. **Add Secrets to Each Scope**

   Using Databricks CLI:
   ```bash
   # Azure OpenAI
   databricks secrets put --scope azure-openai --key endpoint
   databricks secrets put --scope azure-openai --key api-key

   # Azure AI Search
   databricks secrets put --scope azure-search --key endpoint
   databricks secrets put --scope azure-search --key admin-key

   # Azure Storage
   databricks secrets put --scope azure-storage --key account-name
   databricks secrets put --scope azure-storage --key account-key
   ```

3. **Mount ADLS Gen2 Storage**

   Run the setup notebook:
   ```
   notebooks/00_setup/00_azure_environment_setup.py
   ```

   This creates:
   ```
   /mnt/datalake/bronze/   ← Raw data
   /mnt/datalake/silver/   ← Processed data
   /mnt/datalake/gold/     ← Final data
   ```

### Expected Output

```
✓ Secret scopes created
✓ Storage mounted at /mnt/datalake/
✓ Bronze/Silver/Gold directories created
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Secret scope not found" | Create scope in Databricks UI first |
| "Mount failed" | Check storage account key is correct |
| "Access denied" | Ensure Databricks has Storage Blob Data Contributor role |

---

## Step 2: Email Ingestion (PST Files)

### What We're Doing
Extracting emails, attachments, and metadata from Outlook PST archives.

### Why
- **35 years of knowledge**: Emails contain institutional memory
- **Threading**: Group conversations together for context
- **Attachments**: Important documents are often attached to emails

### How PST Processing Works

```
PST File
    │
    ├── Folder 1 (Inbox)
    │   ├── Email 1
    │   │   ├── Headers (from, to, date, subject)
    │   │   ├── Body (text content)
    │   │   └── Attachments (files)
    │   └── Email 2...
    │
    └── Folder 2 (Sent Items)
        └── ...
```

### Instructions

1. **Upload PST Files to Bronze Layer**

   ```python
   # Using Azure Storage Explorer or azcopy
   azcopy copy "local/path/*.pst" "https://<account>.blob.core.windows.net/datalake/bronze/pst/"
   ```

2. **Run the Email Ingestion Notebook**

   ```
   notebooks/01_ingestion/01_pst_email_ingestion.py
   ```

3. **What the Notebook Does**

   ```python
   # Step 1: List all PST files
   pst_files = dbutils.fs.ls("/mnt/datalake/bronze/pst/")

   # Step 2: For each PST file
   for pst_file in pst_files:
       # Extract emails using our PSTLoader
       loader = PSTLoader(pst_file.path)
       emails = loader.load()

       # Step 3: Extract metadata
       for email in emails:
           metadata = {
               "message_id": email.message_id,
               "sender": email.sender,
               "recipients": email.recipients,
               "subject": email.subject,
               "date": email.date,
               "has_attachments": len(email.attachments) > 0
           }

       # Step 4: Build email threads
       threader = EmailThreader()
       threads = threader.build_threads(emails)

       # Step 5: Save to Delta Lake
       emails_df.write.format("delta").save("/mnt/datalake/bronze/emails")
   ```

4. **Understanding Email Threading**

   Why threading matters:
   ```
   Without Threading:
   - Email 1: "Let's discuss the project"
   - Email 2: "RE: Let's discuss the project"
   - Email 3: "RE: RE: Let's discuss the project"
   (Treated as 3 separate documents - loses context!)

   With Threading:
   - Thread 1: [Email 1 → Email 2 → Email 3]
   (Full conversation preserved - better for RAG!)
   ```

### Expected Output

```
Bronze Layer:
├── /mnt/datalake/bronze/emails/
│   ├── part-00000.parquet
│   └── _delta_log/
├── /mnt/datalake/bronze/attachments/
└── /mnt/datalake/bronze/email_threads/

Statistics:
- Total emails processed: ~500,000
- Total threads: ~150,000
- Total attachments: ~50,000
```

### Data Schema

| Column | Type | Description |
|--------|------|-------------|
| message_id | string | Unique email identifier |
| thread_id | string | Conversation thread ID |
| sender | string | Email sender |
| recipients | array | To/CC recipients |
| subject | string | Email subject |
| body | string | Email content |
| date | timestamp | Send date |
| attachments | array | Attachment filenames |
| pst_source | string | Source PST file |

---

## Step 3: Document Ingestion

### What We're Doing
Loading and extracting text from various document formats.

### Why
- **Multiple Formats**: Enterprise uses PDF, Word, Excel, PowerPoint
- **Preserve Structure**: Tables, headers, sections matter
- **Extract Everything**: Images, charts, embedded objects

### How Different Formats Are Processed

| Format | Library | What's Extracted |
|--------|---------|------------------|
| PDF | PyPDF2 / pdfplumber | Text, tables, metadata |
| DOCX | python-docx | Text, tables, headers, styles |
| XLSX | openpyxl | All sheets, cells, formulas |
| PPTX | python-pptx | Slides, speaker notes, text boxes |
| MSG | extract-msg | Email content (like PST) |

### Instructions

1. **Upload Documents to Bronze Layer**

   Organize by type:
   ```
   /mnt/datalake/bronze/documents/
   ├── pdf/
   ├── docx/
   ├── xlsx/
   ├── pptx/
   └── msg/
   ```

2. **Run the Document Ingestion Notebook**

   ```
   notebooks/01_ingestion/02_document_ingestion.py
   ```

3. **What the Notebook Does**

   ```python
   # Use DocumentLoaderFactory to handle all types
   from loaders.document_loader import DocumentLoaderFactory

   factory = DocumentLoaderFactory()

   # Process each document
   for file_path in document_files:
       # Factory selects correct loader based on extension
       loader = factory.get_loader(file_path)

       # Extract content
       documents = loader.load()

       # Each document has:
       # - page_content: The extracted text
       # - metadata: File info, page numbers, etc.
   ```

4. **Handling Large Files**

   ```python
   # For very large PDFs (100+ pages)
   # We process in batches to avoid memory issues

   loader = PDFLoader(file_path, batch_size=20)  # 20 pages at a time

   for batch in loader.load_batches():
       # Process and save each batch
       save_to_delta(batch)
   ```

### Expected Output

```
Bronze Layer:
├── /mnt/datalake/bronze/documents/
│   ├── pdf_extracted/
│   ├── docx_extracted/
│   ├── xlsx_extracted/
│   └── pptx_extracted/

Statistics:
- PDFs processed: 10,000
- DOCX processed: 5,000
- XLSX processed: 2,000
- PPTX processed: 1,000
- Total pages: ~500,000
```

### Data Schema

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Unique document ID |
| file_name | string | Original filename |
| file_type | string | pdf/docx/xlsx/pptx |
| content | string | Extracted text |
| page_number | int | Page/sheet/slide number |
| file_path | string | Source path |
| file_size | long | Size in bytes |
| created_date | timestamp | File creation date |
| modified_date | timestamp | Last modified date |

---

## Step 4: Language Detection

### What We're Doing
Identifying whether each document is English or Dutch.

### Why
- **Different NLP Models**: spaCy has separate models for EN and NL
- **Better Accuracy**: Language-specific models perform better
- **Multilingual Enterprise**: Global company = multiple languages

### How Language Detection Works

```python
# We use langdetect library
from langdetect import detect, detect_langs

text = "This is an English document about the project."
language = detect(text)  # Returns 'en'

text = "Dit is een Nederlands document over het project."
language = detect(text)  # Returns 'nl'

# For confidence scores
probs = detect_langs(text)  # Returns [nl:0.95, en:0.05]
```

### Instructions

1. **Run the Language Detection Notebook**

   ```
   notebooks/01_ingestion/03_language_detection.py
   ```

2. **What the Notebook Does**

   ```python
   from langdetect import detect, LangDetectException

   def detect_language(text):
       """Detect language with fallback."""
       if not text or len(text) < 20:
           return "unknown"

       try:
           lang = detect(text)
           # We only care about EN and NL
           if lang in ['en', 'nl']:
               return lang
           else:
               return 'other'
       except LangDetectException:
           return "unknown"

   # Apply to all documents
   documents_df = documents_df.withColumn(
       "language",
       detect_language_udf(col("content"))
   )
   ```

3. **Handling Mixed Language Documents**

   Some documents have both English and Dutch:
   ```python
   def detect_primary_language(text):
       """Detect primary language for mixed documents."""
       # Split into paragraphs
       paragraphs = text.split('\n\n')

       # Detect language for each paragraph
       langs = [detect(p) for p in paragraphs if len(p) > 50]

       # Return most common language
       return max(set(langs), key=langs.count)
   ```

### Expected Output

```
Language Distribution:
- English: 70% (350,000 documents)
- Dutch: 25% (125,000 documents)
- Other: 3% (15,000 documents)
- Unknown: 2% (10,000 documents)

Updated Bronze Tables:
- All document tables now have 'language' column
```

### Data Update

| Column | Type | Description |
|--------|------|-------------|
| language | string | 'en', 'nl', 'other', 'unknown' |
| language_confidence | float | Detection confidence (0-1) |

---

## Phase 1 Checklist

Before moving to Phase 2, verify:

- [ ] Azure environment is set up
  - [ ] Databricks secrets configured
  - [ ] Storage mounted at `/mnt/datalake/`
  - [ ] Bronze/Silver/Gold directories exist

- [ ] Emails are ingested
  - [ ] All PST files processed
  - [ ] Email threads built
  - [ ] Attachments extracted

- [ ] Documents are ingested
  - [ ] PDFs processed
  - [ ] DOCX files processed
  - [ ] XLSX files processed
  - [ ] PPTX files processed

- [ ] Languages are detected
  - [ ] All documents have language tag
  - [ ] EN/NL distribution looks reasonable

---

## Verification Queries

Run these in Databricks to verify Phase 1:

```python
# Check email count
emails_df = spark.read.format("delta").load("/mnt/datalake/bronze/emails")
print(f"Total emails: {emails_df.count()}")

# Check document count
docs_df = spark.read.format("delta").load("/mnt/datalake/bronze/documents")
print(f"Total documents: {docs_df.count()}")

# Check language distribution
docs_df.groupBy("language").count().show()

# Check for any failed extractions
docs_df.filter(col("content").isNull()).count()
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| PST file won't open | Corrupted or encrypted | Try `scanpst.exe` to repair |
| PDF text is empty | Scanned PDF (image) | Need OCR (Azure Document Intelligence) |
| DOCX extraction fails | Password protected | Remove password or skip file |
| Language detection wrong | Very short text | Require minimum 50 characters |
| Out of memory | Large files | Reduce batch size, use streaming |

---

## What's Next

In **Phase 2: NLP Processing**, we will:
1. Split documents into semantic chunks
2. Extract named entities (people, organizations, etc.)
3. Anonymize personal information (PII)
4. Generate summaries

---

*Phase 1 Complete! Proceed to [Phase 2: NLP Processing](./PHASE_2_NLP_PROCESSING.md)*
