# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Azure Environment Setup
# MAGIC
# MAGIC **Phase 1: Setup & Ingestion | Week 1**
# MAGIC
# MAGIC This notebook configures the Azure Databricks environment for the Knowledge Structuring Pipeline.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Validate Azure service connections
# MAGIC - Configure Databricks secrets
# MAGIC - Set up Delta Lake storage paths
# MAGIC - Test Azure OpenAI connectivity
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Azure Databricks workspace deployed
# MAGIC - Azure Key Vault with secrets configured
# MAGIC - Azure OpenAI service provisioned with gpt-4o deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Required Libraries
# MAGIC
# MAGIC Run this cell to install dependencies on the cluster.

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai langchain-community azure-identity azure-storage-file-datalake azure-search-documents openai tiktoken python-dotenv langdetect

# COMMAND ----------

# Restart Python to pick up new packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Databricks Secrets
# MAGIC
# MAGIC Secrets should be stored in Azure Key Vault and linked to Databricks.
# MAGIC
# MAGIC **Setup steps (one-time via Databricks CLI):**
# MAGIC ```bash
# MAGIC # Create secret scope linked to Key Vault
# MAGIC databricks secrets create-scope --scope azure-openai --scope-backend-type AZURE_KEYVAULT \
# MAGIC   --resource-id /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.KeyVault/vaults/{kv-name} \
# MAGIC   --dns-name https://{kv-name}.vault.azure.net/
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate Secrets Access

# COMMAND ----------

def validate_secrets():
    """Validate that all required secrets are accessible."""
    required_secrets = [
        ("azure-openai", "endpoint"),
        ("azure-openai", "api-key"),
        ("azure-storage", "account-name"),
        ("azure-storage", "container-name"),
    ]

    results = []
    for scope, key in required_secrets:
        try:
            value = dbutils.secrets.get(scope=scope, key=key)
            status = "OK" if value else "EMPTY"
        except Exception as e:
            status = f"ERROR: {str(e)[:50]}"
        results.append({"scope": scope, "key": key, "status": status})

    return results

# Display validation results
try:
    results = validate_secrets()
    display(spark.createDataFrame(results))
except NameError:
    print("Running outside Databricks - using environment variables instead")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure Storage Paths (Delta Lake Medallion Architecture)

# COMMAND ----------

# Storage configuration
STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")

# Base path for ADLS Gen2
BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Medallion architecture paths
PATHS = {
    "raw": f"{BASE_PATH}/raw",           # Original PST, PDF, DOCX files
    "bronze": f"{BASE_PATH}/bronze",     # Extracted text, minimal processing
    "silver": f"{BASE_PATH}/silver",     # Cleaned, chunked, NER-tagged
    "gold": f"{BASE_PATH}/gold",         # Final embeddings, graph, summaries
}

# Display paths
for zone, path in PATHS.items():
    print(f"{zone.upper():8} -> {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Storage Directories

# COMMAND ----------

# Create directory structure in ADLS
for zone, path in PATHS.items():
    try:
        dbutils.fs.mkdirs(path)
        print(f"Created: {zone}")
    except Exception as e:
        print(f"Exists or error for {zone}: {e}")

# Verify structure
display(dbutils.fs.ls(BASE_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Azure OpenAI Connection

# COMMAND ----------

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Get credentials
OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="gpt-4o",
    temperature=0
)

# Test connection
response = llm.invoke("Say 'Azure OpenAI connection successful!' and nothing else.")
print(f"LLM Response: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Embeddings Model

# COMMAND ----------

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
)

# Test embedding
test_text = "Knowledge structuring from enterprise documents"
embedding_vector = embeddings.embed_query(test_text)

print(f"Embedding dimensions: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Configuration as Widgets (for downstream notebooks)

# COMMAND ----------

# Create widgets for configuration (accessible in other notebooks via %run)
dbutils.widgets.text("storage_account", STORAGE_ACCOUNT, "Storage Account")
dbutils.widgets.text("container", CONTAINER, "Container")
dbutils.widgets.text("bronze_path", PATHS["bronze"], "Bronze Path")
dbutils.widgets.text("silver_path", PATHS["silver"], "Silver Path")
dbutils.widgets.text("gold_path", PATHS["gold"], "Gold Path")

print("Configuration widgets created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Environment Summary

# COMMAND ----------

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║           AZURE ENVIRONMENT CONFIGURATION COMPLETE               ║
╠══════════════════════════════════════════════════════════════════╣
║  Storage Account : {STORAGE_ACCOUNT:<43} ║
║  Container       : {CONTAINER:<43} ║
║  OpenAI Endpoint : {OPENAI_ENDPOINT[:40]:<43} ║
║  Embedding Dims  : 3072                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  MEDALLION PATHS:                                                ║
║  • Raw    : /raw     (original files)                            ║
║  • Bronze : /bronze  (extracted text)                            ║
║  • Silver : /silver  (processed, chunked)                        ║
║  • Gold   : /gold    (embeddings, graph)                         ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `01_ingestion/01_pst_email_loader.py` to start PST ingestion
# MAGIC 2. Run `01_ingestion/02_document_loader.py` for PDF, DOCX, XLSX files
# MAGIC 3. Proceed to language detection and chunking
