# Deploy Gradio App on Azure Container App

Hosts the query/chat UI with full access to Cosmos DB, AI Search, ADLS, and Azure OpenAI.

**Cost:** ~$5-10/month (scales to zero when not in use).

---

## Step 1: Create Azure Container Registry

```bash
az acr create \
  --resource-group rg-thesis-pipeline \
  --name thesispipelineacr \
  --sku Basic
```

Enable admin access:

```bash
az acr update --name thesispipelineacr --admin-enabled true
```

Get credentials (you'll need these later):

```bash
az acr credential show --name thesispipelineacr
```

---

## Step 2: Build and Push Docker Image

From your project root:

```bash
cd /Users/rafiq/Learn_KU/Thesis/pipeline/dev/mater_thesis

# Login to ACR
az acr login --name thesispipelineacr

# Build and push
docker build -t thesispipelineacr.azurecr.io/email-kg-app:latest .
docker push thesispipelineacr.azurecr.io/email-kg-app:latest
```

**Alternative — build directly in ACR (no local Docker needed):**

```bash
az acr build \
  --registry thesispipelineacr \
  --image email-kg-app:latest \
  .
```

---

## Step 3: Create Container App Environment

```bash
az containerapp env create \
  --name thesis-app-env \
  --resource-group rg-thesis-pipeline \
  --location eastus2
```

---

## Step 4: Deploy Container App

```bash
az containerapp create \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --environment thesis-app-env \
  --image thesispipelineacr.azurecr.io/email-kg-app:latest \
  --registry-server thesispipelineacr.azurecr.io \
  --registry-username thesispipelineacr \
  --registry-password "<acr-password-from-step-1>" \
  --target-port 7861 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 2 \
  --memory 4Gi \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="<your-endpoint>" \
    AZURE_OPENAI_API_KEY="<your-key>" \
    AZURE_OPENAI_DEPLOYMENT="structexp-4o" \
    AZURE_OPENAI_API_VERSION="2025-01-01-preview" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small" \
    COSMOS_GREMLIN_ENDPOINT="wss://<your-account>.gremlin.cosmos.azure.com:443/" \
    COSMOS_GREMLIN_KEY="<your-gremlin-key>" \
    COSMOS_DATABASE="email-kg" \
    COSMOS_GRAPH="knowledge-graph" \
    COSMOS_NOSQL_ENDPOINT="https://<your-account>.documents.azure.com:443/" \
    COSMOS_NOSQL_KEY="<your-nosql-key>" \
    AZURE_SEARCH_ENDPOINT="https://<your-search>.search.windows.net" \
    AZURE_SEARCH_API_KEY="<your-search-key>" \
    PIPELINE_MODE="llm"
```

---

## Step 5: Get Your App URL

```bash
az containerapp show \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

This gives you a public URL like: `https://email-kg-app.politeground-xxxxx.eastus2.azurecontainerapps.io`

Open it in your browser — your Gradio app is live.

---

## Step 6: Update the App After Code Changes

After modifying `src/` locally:

```bash
# Rebuild and push
docker build -t thesispipelineacr.azurecr.io/email-kg-app:latest .
docker push thesispipelineacr.azurecr.io/email-kg-app:latest

# Restart the container app to pick up the new image
az containerapp update \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --image thesispipelineacr.azurecr.io/email-kg-app:latest
```

---

## How Data Flows

The Gradio app reads all query-time data from Azure services — **no local Silver/Gold files needed**:

- **Azure AI Search** — vector + keyword hybrid search over chunk embeddings and summaries
- **Cosmos DB NoSQL** — chunk details, thread summaries, community summaries (by ID or thread)
- **Cosmos DB Gremlin** — knowledge graph traversals, entity lookups, PathRAG paths

When `COSMOS_GREMLIN_ENDPOINT` or `COSMOS_NOSQL_ENDPOINT` env vars are set, the retrieval layer automatically switches to DB reads. No ADLS sync or local file I/O happens at query time.

The `--gold` and `--silver` CLI args still exist for local development but are not needed in the Container App deployment.

---

## Secure Secrets with Key Vault (Recommended)

Instead of passing secrets as plain env vars:

### 1. Create Managed Identity

```bash
az containerapp identity assign \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --system-assigned
```

### 2. Grant Key Vault Access

```bash
# Get the identity principal ID
PRINCIPAL_ID=$(az containerapp identity show \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --query "principalId" --output tsv)

# Grant access to Key Vault secrets
az keyvault set-policy \
  --name thesis-kv \
  --object-id $PRINCIPAL_ID \
  --secret-permissions get list
```

### 3. Reference Secrets in Container App

```bash
az containerapp update \
  --name email-kg-app \
  --resource-group rg-thesis-pipeline \
  --set-env-vars \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    ADLS_STORAGE_KEY=secretref:adls-storage-key \
  --secrets \
    azure-openai-key=keyvaultref:https://thesis-kv.vault.azure.net/secrets/azure-openai-key,identityref:system \
    adls-storage-key=keyvaultref:https://thesis-kv.vault.azure.net/secrets/adls-storage-key,identityref:system
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Container App (min-replicas=0, scales to zero) | $0 when idle, ~$0.05/hour when active |
| Container Registry (Basic) | ~$5/month |
| Estimated monthly (light thesis use) | ~$5-10 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App won't start | Check logs: `az containerapp logs show --name email-kg-app --resource-group rg-thesis-pipeline` |
| Can't reach Cosmos DB / AI Search | Ensure Container App and Azure services are in the same region. Check NSG/firewall rules. |
| Slow first request | Scale-to-zero means cold start (~30s). Set `--min-replicas 1` to keep warm ($$$). |
| Out of memory | Increase `--memory 8Gi` if Gold layer is large |
| Image too large | Add `.dockerignore` to exclude `data/`, `.venv/`, `.git/`, `pathrag_reference/` |
