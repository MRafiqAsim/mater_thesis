# Azure Synapse Deployment Guide

Step-by-step guide to deploy the Email Knowledge Graph pipeline on Azure Synapse with ADLS Gen2.

**Prerequisites**: Azure subscription, Azure CLI installed, your existing GPT-4o key and endpoint.

---

## Step 1: Create Azure Resources

### 1.1 Create Resource Group

Go to Azure Portal → Resource Groups → Create.

```
Name:     rg-thesis-pipeline
Region:   East US 2  (same region as your Azure OpenAI)
```

### 1.2 Create ADLS Gen2 Storage Account

Go to Azure Portal → Storage Accounts → Create.

```
Resource Group:   rg-thesis-pipeline
Name:             thesispipelinestore    (must be globally unique, lowercase, no hyphens)
Region:           East US 2
Performance:      Standard
Redundancy:       LRS (cheapest, fine for thesis)
Advanced tab:     ✅ Enable hierarchical namespace   ← THIS IS CRITICAL for ADLS Gen2
```

Click **Create**.

### 1.3 Create Container and Folder Structure

After the storage account is created:

1. Go to **Storage Account → Containers → + Container**
2. Name: `pipeline-data`, Access level: Private
3. Click into `pipeline-data` and create these folders using **+ Add Directory**:

```
input/
input/source/
input/processing/
input/processed/
bronze/
bronze/emails/
bronze/documents/
bronze/attachments/
silver_llm/
gold_llm/
config/
```

### 1.4 Get Storage Account Key

Go to **Storage Account → Access keys → key1 → Show → Copy**.

Save this — you'll need it as `ADLS_STORAGE_KEY`.

### 1.5 Create Synapse Workspace

Go to Azure Portal → Azure Synapse Analytics → Create.

```
Resource Group:          rg-thesis-pipeline
Workspace name:          thesis-synapse-ws
Region:                  East US 2
Data Lake Storage Gen2:  Select your storage account (thesispipelinestore)
File system:             pipeline-data
```

Set the SQL admin credentials (you won't use SQL, but it's required).

Click **Create**. Wait ~5 minutes.

### 1.6 Create Apache Spark Pool

After Synapse workspace is ready:

1. Go to **Synapse workspace → Apache Spark pools → + New**

```
Name:           MediumPool
Node size:      Medium (8 vCores, 64 GB)
Autoscale:      ✅ Enabled
Min nodes:      3
Max nodes:      6
Auto-pause:     ✅ Enabled, 15 minutes
Spark version:  3.4 (latest available)
```

2. Click **Create**.

### 1.7 Create Cosmos DB Account (Gremlin API)

Go to Azure Portal → Azure Cosmos DB → Create → **Azure Cosmos DB for Apache Gremlin**.

```
Resource Group:   rg-thesis-pipeline
Account name:     thesis-cosmos-gremlin    (globally unique)
Region:           East US 2
Capacity mode:    Serverless               ← cheapest for thesis
```

After creation:

1. Go to **Cosmos DB account → Data Explorer → New Database**
   - Database id: `email-kg`
2. Inside `email-kg`, click **New Graph**
   - Graph id: `knowledge-graph`
   - Partition key: `/node_type`
3. Go to **Keys** → copy **URI** and **PRIMARY KEY**
   - Save as `COSMOS_GREMLIN_ENDPOINT` (use the Gremlin endpoint, e.g., `wss://thesis-cosmos-gremlin.gremlin.cosmos.azure.com:443/`)
   - Save as `COSMOS_GREMLIN_KEY`

### 1.8 Create Cosmos DB Account (NoSQL API)

Go to Azure Portal → Azure Cosmos DB → Create → **Azure Cosmos DB for NoSQL**.

```
Resource Group:   rg-thesis-pipeline
Account name:     thesis-cosmos-nosql      (globally unique)
Region:           East US 2
Capacity mode:    Serverless
```

After creation:

1. Go to **Cosmos DB account → Data Explorer → New Database**
   - Database id: `email-kg`
2. Inside `email-kg`, create 3 containers:

| Container | Partition Key |
|-----------|--------------|
| `chunks` | `/thread_id` |
| `communities` | `/level` |
| `thread_summaries` | `/thread_id` |

3. Go to **Keys** → copy **URI** and **PRIMARY KEY**
   - Save as `COSMOS_NOSQL_ENDPOINT`
   - Save as `COSMOS_NOSQL_KEY`

> **Alternative**: You can use a single Cosmos DB account with multi-model if available in your region. The guide assumes separate accounts for clarity.

### 1.9 Create Azure AI Search

Go to Azure Portal → Azure AI Search → Create.

```
Resource Group:   rg-thesis-pipeline
Service name:     thesis-ai-search        (globally unique)
Region:           East US 2
Pricing tier:     Free (50 MB, fine for thesis)   or Basic ($75/month for larger data)
```

After creation:

1. Go to **AI Search → Keys** → copy **Primary admin key**
   - Save as `AZURE_SEARCH_API_KEY`
2. Copy the **URL** from the Overview page (e.g., `https://thesis-ai-search.search.windows.net`)
   - Save as `AZURE_SEARCH_ENDPOINT`

The search index is created automatically by the `03_gold_indexing` notebook when it runs the embedding generator.

---

## Step 2: Upload Project Code to Synapse

### 2.1 Open Synapse Studio

Go to **Synapse workspace → Overview → Open Synapse Studio** (or go to `https://web.azuresynapse.net`).

### 2.2 Upload Source Code as Workspace Package

Your `src/` code needs to be packaged as a `.whl` or `.zip` so Synapse notebooks can import it.

**On your local machine**, run:

```bash
cd /Users/rafiq/Learn_KU/Thesis/pipeline/dev/mater_thesis

# Create a zip of the source code
zip -r src_package.zip src/ config/prompts.json -x "src/__pycache__/*" "src/**/__pycache__/*"
```

Then upload to Synapse:

1. In Synapse Studio → **Manage → Workspace packages → + Upload**
2. Upload `src_package.zip`

**Alternative (easier)**: Upload the entire project to ADLS and add it to `sys.path` in notebooks:

```bash
# On your local machine — upload source to ADLS
az storage fs directory upload \
  --account-name thesispipelinestore \
  --file-system pipeline-data \
  --source ./src \
  --destination-path code/src \
  --recursive
```

```bash
# Also upload config
az storage blob upload \
  --account-name thesispipelinestore \
  --container-name pipeline-data \
  --name config/prompts.json \
  --file ./config/prompts.json
```

### 2.3 Upload .env Configuration

Upload your prompts configuration:

```bash
az storage blob upload \
  --account-name thesispipelinestore \
  --container-name pipeline-data \
  --name config/prompts.json \
  --file ./config/prompts.json
```

---

## Step 3: Configure Synapse Environment

### 3.1 Create Environment (requirements)

In Synapse Studio → **Manage → Apache Spark pools → MediumPool → Packages**.

Click **+ Upload** and upload a `requirements.txt`:

```txt
tiktoken==0.7.0
presidio-analyzer==2.2.355
presidio-anonymizer==2.2.355
spacy==3.7.5
openai==1.52.0
python-dotenv==1.0.1
networkx==3.3
graspologic==3.4.1
pydantic==2.9.2
httpx==0.27.2
azure-storage-file-datalake==12.17.0
azure-identity==1.19.0
azure-search-documents==11.6.0b5
azure-cosmos==4.7.0
gremlinpython==3.7.2
sentence-transformers==3.1.1
gradio==5.8.0
langdetect==1.0.9
ftfy==6.3.1
```

After upload, click **Apply**. Wait for the pool to restart (~5 min).

### 3.2 Download spaCy Models

This must be done once per Spark pool restart. We handle this in notebook `00_setup`.

---

## Step 4: Create Notebooks in Synapse Studio

### 4.1 Create Notebook: 00_setup

1. In Synapse Studio → **Develop → + → Notebook**
2. Name it `00_setup`
3. Attach to `MediumPool`
4. Copy-paste the content from `synapse/notebooks/00_setup.py`
5. **Edit the configuration cell** — replace placeholders:

```python
os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-endpoint>"
os.environ["AZURE_OPENAI_API_KEY"] = "<your-key>"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "structexp-4o"
os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = "text-embedding-3-small"

os.environ["ADLS_STORAGE_ACCOUNT"] = "<your-storage-account>"
os.environ["ADLS_CONTAINER"] = "pipeline-data"

# Cosmos DB Gremlin
os.environ["COSMOS_GREMLIN_ENDPOINT"] = "wss://<your-account>.gremlin.cosmos.azure.com:443/"
os.environ["COSMOS_GREMLIN_KEY"] = "<your-gremlin-key>"
os.environ["COSMOS_DATABASE"] = "email-kg"
os.environ["COSMOS_GRAPH"] = "knowledge-graph"

# Cosmos DB NoSQL
os.environ["COSMOS_NOSQL_ENDPOINT"] = "https://<your-account>.documents.azure.com:443/"
os.environ["COSMOS_NOSQL_KEY"] = "<your-nosql-key>"

# Azure AI Search
os.environ["AZURE_SEARCH_ENDPOINT"] = "https://<your-search>.search.windows.net"
os.environ["AZURE_SEARCH_API_KEY"] = "<your-search-key>"
```

6. **Add a first cell** to make code importable:

```python
# Cell 1: Mount code from ADLS
import subprocess, os

# Download source code from ADLS to local
subprocess.run([
    "az", "storage", "fs", "directory", "download",
    "--account-name", "thesispipelinestore",
    "--file-system", "pipeline-data",
    "--source-path", "code/src",
    "--destination-path", "/tmp/pipeline_code/",
    "--recursive"
], check=True)

# Add to Python path
import sys
sys.path.insert(0, "/tmp/pipeline_code")
```

**Or simpler**: Use the Synapse ADLS mount that's already available:

```python
# Cell 1: Add code to path
import sys
sys.path.insert(0, "/synapse/pipeline-data/code")
# This works because Synapse auto-mounts the linked ADLS
```

7. Click **Publish** to save.

### 4.2 Create Notebook: 01_ingestion

1. **Develop → + → Notebook**, name: `01_ingestion`
2. Attach to `MediumPool`
3. Copy-paste from `synapse/notebooks/01_ingestion.py`
4. **Add code path cell at the top** (same as 00_setup Cell 1)
5. **Add env vars cell** (same values as 00_setup)
6. **Publish**

### 4.3 Create Notebook: 02_thread_processing

1. **Develop → + → Notebook**, name: `02_thread_processing`
2. Attach to `MediumPool`
3. Copy-paste from `synapse/notebooks/02_thread_processing.py`
4. Add code path + env vars cells at top
5. **Publish**

### 4.4 Create Notebook: 03_gold_indexing

1. **Develop → + → Notebook**, name: `03_gold_indexing`
2. Attach to `MediumPool`
3. Copy-paste from `synapse/notebooks/03_gold_indexing.py`
4. Add code path + env vars cells at top
5. **Publish**

### 4.5 Create Notebook: 04_query_service

1. **Develop → + → Notebook**, name: `04_query_service`
2. Attach to `MediumPool`
3. Copy-paste from `synapse/notebooks/04_query_service.py`
4. Add code path + env vars cells at top
5. **Publish**

---

## Step 5: Test Each Notebook Manually

Run them one by one to verify everything works before setting up automation.

### 5.1 Run 00_setup

1. Open `00_setup` notebook
2. Click **Run All**
3. Verify:
   - All pip installs succeed
   - spaCy models download
   - ADLS connectivity test passes (folder structure created)
   - Azure OpenAI test returns "OK"

### 5.2 Upload Test Data

Upload a PST file or some documents to `input/source/`:

```bash
# From your local machine
az storage blob upload \
  --account-name thesispipelinestore \
  --container-name pipeline-data \
  --name input/source/test_emails.pst \
  --file ./data/input/your_pst_file.pst
```

Or upload via Azure Portal: **Storage Account → Containers → pipeline-data → input → source → Upload**.

### 5.3 Run 01_ingestion

1. Open `01_ingestion`
2. Run All
3. Verify:
   - Files detected in `input/source/`
   - Moved to `input/processing/`
   - Emails extracted to `bronze/emails/`
   - Files moved to `input/processed/{timestamp}/`
   - Check Azure Portal → pipeline-data → bronze/emails/ — you should see JSON files

### 5.4 Run 02_thread_processing

1. Open `02_thread_processing`
2. Run All
3. Verify:
   - Bronze emails downloaded to local temp
   - Identity registry built
   - Silver chunks created
   - Uploaded to `silver_llm/not_personal/email_chunks/`
   - Check Azure Portal → pipeline-data → silver_llm/ — you should see JSON chunks

### 5.5 Run 03_gold_indexing

1. Open `03_gold_indexing`
2. Run All
3. Verify:
   - Knowledge graph built (nodes + edges printed)
   - Communities detected
   - Paths indexed
   - Embeddings generated
   - Check Azure Portal → pipeline-data → gold_llm/ — graph, communities, paths, embeddings folders

### 5.6 Run 05_cosmos_upload

1. Create notebook `05_cosmos_upload` in Synapse Studio (same steps as 4.1-4.5)
2. Copy-paste from `synapse/notebooks/05_cosmos_upload.py`
3. Add code path + env vars cells at top (include Cosmos DB vars)
4. Run All
5. Verify:
   - NoSQL containers created (chunks, communities, thread_summaries)
   - Nodes and edges uploaded to Gremlin
   - Chunks and summaries uploaded to NoSQL
   - Check Azure Portal → Cosmos DB → Data Explorer — you should see data

### 5.7 Run 04_query_service

1. Open `04_query_service`
2. Run the batch evaluation cells
3. Verify queries return answers with confidence scores

---

## Step 6: Create Synapse Pipeline

### 6.1 Create Pipeline

1. In Synapse Studio → **Integrate → + → Pipeline**
2. Name: `EmailKnowledgePipeline`

### 6.2 Add Activities

Drag from the Activities panel on the left:

**Activity 1: Ingestion**
1. Drag **Synapse notebook** activity onto the canvas
2. Name: `01_Ingestion`
3. Settings tab:
   - Notebook: `01_ingestion`
   - Spark pool: `MediumPool`
4. General tab:
   - Timeout: `02:00:00`
   - Retry: `1`

**Activity 2: Thread Processing**
1. Drag another **Synapse notebook** activity
2. Name: `02_Thread_Processing`
3. Settings:
   - Notebook: `02_thread_processing`
   - Spark pool: `MediumPool`
4. General:
   - Timeout: `04:00:00`
   - Retry: `1`
5. **Draw an arrow** from `01_Ingestion` → `02_Thread_Processing` (success dependency)

**Activity 3: Gold Indexing**
1. Drag **Synapse notebook**
2. Name: `03_Gold_Indexing`
3. Settings:
   - Notebook: `03_gold_indexing`
   - Spark pool: `MediumPool`
4. General:
   - Timeout: `02:00:00`
5. **Arrow** from `02_Thread_Processing` → `03_Gold_Indexing`

**Activity 4: Cosmos Upload**
1. Drag **Synapse notebook**
2. Name: `05_Cosmos_Upload`
3. Settings:
   - Notebook: `05_cosmos_upload`
   - Spark pool: `MediumPool`
4. General:
   - Timeout: `02:00:00`
5. **Arrow** from `03_Gold_Indexing` → `05_Cosmos_Upload`

**Activity 5: Evaluation (optional)**
1. Drag **Synapse notebook**
2. Name: `04_Evaluation`
3. Settings:
   - Notebook: `04_query_service`
4. **Arrow** from `05_Cosmos_Upload` → `04_Evaluation`

### 6.3 The pipeline should look like:

```
01_Ingestion → 02_Thread_Processing → 03_Gold_Indexing → 05_Cosmos_Upload → 04_Evaluation
```

### 6.4 Validate and Publish

1. Click **Validate** (top bar) — should show no errors
2. Click **Publish all**
3. Test: Click **Add trigger → Trigger now** to do a manual run
4. Monitor: **Monitor → Pipeline runs** — watch the run progress

---

## Step 7: Set Up Automatic Trigger (Event Grid)

This makes the pipeline run automatically when you drop files in `input/source/`.

### 7.1 Register Event Grid Provider

If not already done:

```bash
az provider register --namespace Microsoft.EventGrid
```

Wait for registration to complete (~2 min):

```bash
az provider show --namespace Microsoft.EventGrid --query "registrationState"
# Should return: "Registered"
```

### 7.2 Create Event Subscription on Storage Account

In Azure Portal:

1. Go to **Storage Account (thesispipelinestore) → Events → + Event Subscription**

```
Name:           new-files-trigger
Event Schema:   Event Grid Schema
Filter to:
  ✅ Blob Created
Subject filter:
  Subject begins with:  /blobServices/default/containers/pipeline-data/blobs/input/source/
  Subject ends with:    (leave empty to catch all file types)
Endpoint type:  Azure Synapse Analytics
Endpoint:       Select your Synapse workspace → EmailKnowledgePipeline
```

2. Click **Create**.

### 7.3 Alternative: Create Trigger in Synapse Studio

1. In Synapse Studio → **Integrate → EmailKnowledgePipeline**
2. Click **Add trigger → New/Edit**
3. **+ New**:

```
Name:    NewFilesTrigger
Type:    Storage events
Account: thesispipelinestore
Container: pipeline-data
Blob path begins with:  input/source/
Blob path ends with:    (leave empty)
Events:  ✅ Blob Created
```

4. Click **OK → Publish**

### 7.4 Test the Trigger

1. Upload a file to `input/source/`:

```bash
az storage blob upload \
  --account-name thesispipelinestore \
  --container-name pipeline-data \
  --name input/source/new_document.pdf \
  --file ./some_local_file.pdf
```

2. Go to **Monitor → Pipeline runs** in Synapse Studio
3. You should see a new run start within ~30 seconds

---

## Step 8: Secure Secrets with Key Vault (Recommended)

Instead of hardcoding API keys in notebooks, use Azure Key Vault.

### 8.1 Create Key Vault

```bash
az keyvault create \
  --name thesis-kv \
  --resource-group rg-thesis-pipeline \
  --location eastus2
```

### 8.2 Store Secrets

```bash
az keyvault secret set --vault-name thesis-kv --name "azure-openai-key" --value "<your-key>"
az keyvault secret set --vault-name thesis-kv --name "adls-storage-key" --value "<your-storage-key>"
az keyvault secret set --vault-name thesis-kv --name "cosmos-gremlin-key" --value "<your-gremlin-key>"
az keyvault secret set --vault-name thesis-kv --name "cosmos-nosql-key" --value "<your-nosql-key>"
az keyvault secret set --vault-name thesis-kv --name "azure-search-key" --value "<your-search-key>"
```

### 8.3 Create Linked Service in Synapse

1. Synapse Studio → **Manage → Linked services → + New**
2. Search **Azure Key Vault**
3. Name: `KeyVaultLS`
4. Select your Key Vault: `thesis-kv`
5. Test connection → **Create**

### 8.4 Use in Notebooks

Replace hardcoded keys with:

```python
from notebookutils import mssparkutils  # Available in Synapse

azure_openai_key = mssparkutils.credentials.getSecret("thesis-kv", "azure-openai-key", "KeyVaultLS")
adls_key = mssparkutils.credentials.getSecret("thesis-kv", "adls-storage-key", "KeyVaultLS")
cosmos_gremlin_key = mssparkutils.credentials.getSecret("thesis-kv", "cosmos-gremlin-key", "KeyVaultLS")
cosmos_nosql_key = mssparkutils.credentials.getSecret("thesis-kv", "cosmos-nosql-key", "KeyVaultLS")
search_key = mssparkutils.credentials.getSecret("thesis-kv", "azure-search-key", "KeyVaultLS")

os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key
os.environ["ADLS_STORAGE_KEY"] = adls_key
os.environ["COSMOS_GREMLIN_KEY"] = cosmos_gremlin_key
os.environ["COSMOS_NOSQL_KEY"] = cosmos_nosql_key
os.environ["AZURE_SEARCH_API_KEY"] = search_key
```

---

## Step 9: Monitor and Debug

### 9.1 Monitor Pipeline Runs

Synapse Studio → **Monitor → Pipeline runs**

- Green checkmark = success
- Red X = failed — click to see which activity failed
- Click activity → **Output** tab to see notebook output and errors

### 9.2 Monitor Spark Applications

Synapse Studio → **Monitor → Apache Spark applications**

- See running/completed notebook executions
- Click for Spark UI, logs, and stdout

### 9.3 Check ADLS Data

Azure Portal → Storage Account → Containers → pipeline-data

Verify files appear in the expected folders after each stage:
- After ingestion: `bronze/emails/*.json`
- After Silver: `silver_llm/not_personal/email_chunks/*.json`
- After Gold: `gold_llm/knowledge_graph/`, `communities/`, `paths/`, `embeddings/`
- Processed inputs: `input/processed/{timestamp}/`

### 9.4 Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Check sys.path cell — ensure code is accessible from ADLS mount or /tmp download |
| `pypff` import error | pypff needs C libraries; install `pypff-compat` via requirements.txt on pool |
| Rate limit (429) on Azure OpenAI | Add `time.sleep(1)` between batches or increase your quota at aka.ms/oai/quotaincrease |
| Spark pool takes long to start | Normal on first run (~5 min). Auto-pause means cold starts. |
| Event Grid trigger not firing | Check Event Subscription in Storage Account → Events → verify Subject filter matches |
| `dbutils.notebook.exit()` error locally | This only works in Synapse/Databricks — ignored when running locally |
| Notebook timeout | Increase timeout in pipeline activity settings (Silver can take 4+ hours for large PSTs) |

---

## Cost Estimation

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| ADLS Gen2 (10 GB) | ~$0.50 |
| Synapse Spark Pool (Medium, auto-pause) | ~$2-5/run (pay per use) |
| Azure OpenAI GPT-4o (Silver + Gold) | ~$5-20 (depends on email volume) |
| Azure OpenAI Embeddings | ~$0.50 |
| Azure AI Search (Free tier) | $0 (or ~$75/month for Basic) |
| Cosmos DB Gremlin (Serverless) | ~$1-5 (pay per RU) |
| Cosmos DB NoSQL (Serverless) | ~$1-3 (pay per RU) |
| Event Grid | ~$0 (free tier covers low volume) |
| Key Vault | ~$0.03/secret/month |
| **Total for thesis** | **~$15-40/month** |

Auto-pause on the Spark pool is critical — without it, a Medium pool runs ~$1,200/month.

---

## Quick Reference

| What | Where |
|------|-------|
| Upload files | Storage Account → pipeline-data → input/source/ |
| Manual pipeline run | Synapse Studio → Integrate → EmailKnowledgePipeline → Trigger now |
| Monitor runs | Synapse Studio → Monitor → Pipeline runs |
| View ADLS data | Storage Account → pipeline-data → bronze/ / silver_llm/ / gold_llm/ |
| View graph data | Cosmos DB Gremlin → Data Explorer → knowledge-graph |
| View chunks/communities | Cosmos DB NoSQL → Data Explorer → email-kg |
| View search index | Azure AI Search → Indexes → knowledge-chunks |
| Notebook logs | Monitor → Apache Spark applications → click run → stdout |
| API keys | Key Vault → thesis-kv → Secrets |
| Change prompts | Upload new `config/prompts.json` to pipeline-data/config/ |
| Update code | Re-upload `src/` to pipeline-data/code/src/ |
