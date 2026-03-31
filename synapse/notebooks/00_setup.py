# Synapse Notebook: 00_setup
# Run once to configure the Synapse Spark pool environment.
# Attach to a Synapse Spark pool (Medium: 8 cores, 56GB recommended).

# %% [markdown]
# # Environment Setup
# Installs dependencies and verifies Azure connectivity.

# %%
# --- Install dependencies ---
%pip install tiktoken pypff-compat presidio-analyzer presidio-anonymizer \
    spacy openai python-dotenv networkx graspologic pydantic httpx \
    azure-storage-file-datalake azure-identity azure-search-documents \
    azure-cosmos gremlinpython sentence-transformers

# %%
# --- Download spaCy models ---
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
subprocess.run(["python", "-m", "spacy", "download", "nl_core_news_sm"], check=True)

# %%
# --- Configuration ---
# Option A: Set via Synapse Linked Service / Key Vault (recommended)
# Option B: Set directly here for testing
import os

# Azure OpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-endpoint>"
os.environ["AZURE_OPENAI_API_KEY"] = "<your-key>"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "structexp-4o"
os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = "text-embedding-3-small"

# ADLS Gen2
os.environ["ADLS_STORAGE_ACCOUNT"] = "<your-storage-account>"
os.environ["ADLS_CONTAINER"] = "pipeline-data"
# os.environ["ADLS_STORAGE_KEY"] = "<key>"  # Or use managed identity (no key needed)

# Cosmos DB Gremlin (knowledge graph)
os.environ["COSMOS_GREMLIN_ENDPOINT"] = "<wss://your-account.gremlin.cosmos.azure.com:443/>"
os.environ["COSMOS_GREMLIN_KEY"] = "<your-gremlin-key>"
os.environ["COSMOS_DATABASE"] = "email-kg"
os.environ["COSMOS_GRAPH"] = "knowledge-graph"

# Cosmos DB NoSQL (chunks, communities, thread summaries)
os.environ["COSMOS_NOSQL_ENDPOINT"] = "<https://your-account.documents.azure.com:443/>"
os.environ["COSMOS_NOSQL_KEY"] = "<your-nosql-key>"

# Azure AI Search
os.environ["AZURE_SEARCH_ENDPOINT"] = "<https://your-search.search.windows.net>"
os.environ["AZURE_SEARCH_API_KEY"] = "<your-search-key>"

# Pipeline
os.environ["PIPELINE_MODE"] = "llm"
os.environ["PROCESS_ATTACHMENTS"] = "true"

# %%
# --- Verify ADLS connectivity ---
from src.storage import ADLSAdapter

adapter = ADLSAdapter()
print(f"Storage: {adapter.storage_account}/{adapter.container}")
print(f"Local mode: {adapter.is_local}")

# Create folder structure if needed
for folder in ["input/source", "input/processing", "input/processed",
               "bronze", "silver_llm", "gold_llm", "config"]:
    adapter._ensure_directory(folder)
    print(f"  ✓ {folder}/")

# %%
# --- Verify Azure OpenAI connectivity ---
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=5,
)
print(f"Azure OpenAI: {response.choices[0].message.content}")

# %%
# --- Upload config files to ADLS ---
import json
from pathlib import Path

# Upload prompts.json
prompts_path = Path("config/prompts.json")
if prompts_path.exists():
    adapter.write_text("config/prompts.json", prompts_path.read_text())
    print("Uploaded config/prompts.json")

print("\n✓ Setup complete. Ready to run pipeline notebooks.")
