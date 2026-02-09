# Phase 3: Vector Index & Basic RAG

**Duration:** Week 5
**Goal:** Create searchable index and baseline question-answering system

---

## Overview

### What We're Building

In this phase, we make data searchable:
1. **Vector Embeddings**: Convert text to numbers that capture meaning
2. **Azure AI Search Index**: Store vectors for fast similarity search
3. **Basic RAG Pipeline**: Ask questions, get answers from documents
4. **Baseline Metrics**: Measure performance to compare against later

### Why This Matters

```
QUESTION: "What was the Q3 revenue?"
         │
         ▼
    ┌──────────────────┐
    │ EMBED QUESTION   │ → Convert to vector [0.1, 0.5, -0.3, ...]
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ SEARCH INDEX     │ → Find similar chunks (HNSW algorithm)
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ RETRIEVE TOP-K   │ → Get 5 most relevant chunks
    └──────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ GENERATE ANSWER  │ → GPT-4o creates answer from chunks
    └──────────────────┘
         │
         ▼
ANSWER: "Q3 revenue was $45 million, a 15% increase from Q2."
```

---

## Prerequisites

### From Phase 2

- [ ] Chunks in `/mnt/datalake/silver/chunks_anonymized/`
- [ ] Summaries in `/mnt/datalake/silver/summaries/`
- [ ] NER results in `/mnt/datalake/silver/ner_results/`

### Azure Resources

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| Azure OpenAI | Embeddings + Generation | text-embedding-3-large, gpt-4o |
| Azure AI Search | Vector storage + Search | Standard S1 (or higher) |

---

## Step 1: Generate Embeddings

### What We're Doing
Converting each chunk's text into a 3072-dimensional vector.

### Why Embeddings Work

```
Text: "The cat sat on the mat"
         │
         ▼
   text-embedding-3-large
         │
         ▼
Vector: [0.12, -0.45, 0.78, ..., 0.33]  (3072 numbers)
```

Similar texts → Similar vectors → Can find by similarity!

```
"The cat sat on the mat"      → [0.12, -0.45, 0.78, ...]
"A feline rested on a rug"    → [0.11, -0.44, 0.77, ...]  ← Very similar!
"The stock market crashed"    → [0.89, 0.23, -0.56, ...]  ← Very different!
```

### Why text-embedding-3-large

| Model | Dimensions | Quality | Cost |
|-------|------------|---------|------|
| text-embedding-ada-002 | 1536 | Good | $ |
| text-embedding-3-small | 1536 | Better | $ |
| **text-embedding-3-large** | 3072 | Best | $$ |

We use the best for thesis quality. In production, you might use smaller.

### Instructions

1. **Run the Embedding Notebook**

   ```
   notebooks/03_vector_index/01_embedding_indexing.py
   ```

2. **Embedding Configuration**

   ```python
   from langchain_openai import AzureOpenAIEmbeddings

   embeddings = AzureOpenAIEmbeddings(
       azure_endpoint=AZURE_OPENAI_ENDPOINT,
       api_key=AZURE_OPENAI_KEY,
       azure_deployment="text-embedding-3-large",
       api_version="2024-02-01",
       # Dimensions: 3072 (default for this model)
   )
   ```

3. **Batch Processing (Important!)**

   ```python
   # DON'T do this (too slow, hits rate limits):
   for chunk in chunks:
       embedding = embeddings.embed_query(chunk.text)

   # DO this (batch processing):
   BATCH_SIZE = 100

   def embed_batch(chunks):
       texts = [c.text for c in chunks]
       vectors = embeddings.embed_documents(texts)
       return vectors

   # Process in batches
   all_embeddings = []
   for i in range(0, len(chunks), BATCH_SIZE):
       batch = chunks[i:i+BATCH_SIZE]
       batch_embeddings = embed_batch(batch)
       all_embeddings.extend(batch_embeddings)

       # Rate limiting
       time.sleep(0.5)  # Adjust based on your quota
   ```

4. **Save Embeddings**

   ```python
   # Add embeddings to chunks dataframe
   chunks_with_embeddings = []
   for chunk, embedding in zip(chunks, all_embeddings):
       chunks_with_embeddings.append({
           "chunk_id": chunk.chunk_id,
           "text": chunk.text,
           "embedding": embedding,  # List of 3072 floats
           "doc_id": chunk.doc_id,
           "metadata": chunk.metadata
       })

   # Save to Gold layer
   embeddings_df = spark.createDataFrame(chunks_with_embeddings)
   embeddings_df.write.format("delta").save("/mnt/datalake/gold/chunks_embedded")
   ```

### Expected Output

```
Chunks embedded: 2,000,000
Embedding dimensions: 3072
Total storage: ~25 GB
Processing time: ~6 hours (with rate limiting)
Estimated cost: ~$100
```

---

## Step 2: Create Azure AI Search Index

### What We're Doing
Creating a search index that can find similar vectors quickly.

### Why HNSW Algorithm

**Problem**: With 2 million vectors, checking all of them for each query is too slow.

**Solution**: HNSW (Hierarchical Navigable Small World)
- Pre-builds a graph structure
- Queries only check ~100-200 vectors instead of 2 million
- Accuracy: ~95% (finds approximately correct results)
- Speed: Milliseconds instead of minutes

```
Without HNSW: O(n) = Check 2,000,000 vectors = 30 seconds
With HNSW:    O(log n) = Check ~150 vectors = 10 milliseconds
```

### Instructions

1. **Create Index Schema**

   ```python
   from azure.search.documents.indexes import SearchIndexClient
   from azure.search.documents.indexes.models import (
       SearchIndex,
       SearchField,
       SearchFieldDataType,
       VectorSearch,
       HnswAlgorithmConfiguration,
       VectorSearchProfile,
   )

   # Define fields
   fields = [
       SearchField(name="id", type=SearchFieldDataType.String, key=True),
       SearchField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
       SearchField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
       SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
       SearchField(name="summary", type=SearchFieldDataType.String, searchable=True),
       SearchField(name="language", type=SearchFieldDataType.String, filterable=True),
       SearchField(name="file_name", type=SearchFieldDataType.String, filterable=True),

       # Vector field for semantic search
       SearchField(
           name="text_vector",
           type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
           searchable=True,
           vector_search_dimensions=3072,
           vector_search_profile_name="my-vector-profile"
       ),
   ]

   # Configure HNSW
   vector_search = VectorSearch(
       algorithms=[
           HnswAlgorithmConfiguration(
               name="my-hnsw-config",
               parameters={
                   "m": 4,              # Connections per node
                   "efConstruction": 400,  # Build-time accuracy
                   "efSearch": 500,     # Search-time accuracy
                   "metric": "cosine"   # Similarity metric
               }
           )
       ],
       profiles=[
           VectorSearchProfile(
               name="my-vector-profile",
               algorithm_configuration_name="my-hnsw-config"
           )
       ]
   )

   # Create index
   index = SearchIndex(
       name="document-chunks",
       fields=fields,
       vector_search=vector_search
   )

   index_client.create_or_update_index(index)
   ```

2. **Understanding HNSW Parameters**

   | Parameter | What It Does | Trade-off |
   |-----------|--------------|-----------|
   | m | Connections per node | Higher = better quality, more memory |
   | efConstruction | Build accuracy | Higher = slower build, better graph |
   | efSearch | Query accuracy | Higher = slower query, better results |

3. **Upload Documents to Index**

   ```python
   from azure.search.documents import SearchClient

   search_client = SearchClient(
       endpoint=AZURE_SEARCH_ENDPOINT,
       index_name="document-chunks",
       credential=AzureKeyCredential(AZURE_SEARCH_KEY)
   )

   # Prepare documents
   documents = []
   for chunk in chunks_with_embeddings:
       documents.append({
           "id": chunk["chunk_id"],
           "chunk_id": chunk["chunk_id"],
           "doc_id": chunk["doc_id"],
           "text": chunk["text"],
           "text_vector": chunk["embedding"],
           "language": chunk["language"],
           "file_name": chunk["file_name"]
       })

   # Upload in batches of 1000
   BATCH_SIZE = 1000
   for i in range(0, len(documents), BATCH_SIZE):
       batch = documents[i:i+BATCH_SIZE]
       search_client.upload_documents(batch)
       print(f"Uploaded {i+BATCH_SIZE}/{len(documents)}")
   ```

### Expected Output

```
Index created: document-chunks
Documents indexed: 2,000,000
Index size: ~30 GB
Indexing time: ~2 hours
```

---

## Step 3: Build Basic RAG Pipeline

### What We're Doing
Creating a question-answering system that retrieves relevant chunks and generates answers.

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                             │
│                                                              │
│  User Question                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │ Query Transform │ → Rewrite query for better retrieval   │
│  └─────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │    Retriever    │ → Find top-k relevant chunks           │
│  │  (Hybrid Search)│    - Vector similarity                 │
│  └─────────────────┘    - Keyword matching (BM25)           │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │ Context Builder │ → Format chunks for LLM                │
│  └─────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │   Generator     │ → GPT-4o generates answer              │
│  │    (GPT-4o)     │    with citations                      │
│  └─────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  Answer with Sources                                         │
└─────────────────────────────────────────────────────────────┘
```

### Instructions

1. **Run the Basic RAG Notebook**

   ```
   notebooks/03_vector_index/02_basic_rag.py
   ```

2. **Initialize RAG Chain**

   ```python
   from retrieval.rag_chain import RAGChain, RAGConfig

   config = RAGConfig(
       # Retrieval settings
       top_k=5,                    # Number of chunks to retrieve
       score_threshold=0.7,       # Minimum similarity score
       use_hybrid_search=True,    # Combine vector + keyword

       # Generation settings
       model_deployment="gpt-4o",
       temperature=0.1,           # Low for factual answers
       max_tokens=1000,

       # Citation settings
       include_citations=True,
       citation_format="[Source: {filename}]"
   )

   rag = RAGChain(
       azure_endpoint=AZURE_OPENAI_ENDPOINT,
       api_key=AZURE_OPENAI_KEY,
       search_endpoint=AZURE_SEARCH_ENDPOINT,
       search_key=AZURE_SEARCH_KEY,
       index_name="document-chunks",
       config=config
   )
   ```

3. **Hybrid Search Explained**

   Why combine vector and keyword search?

   ```
   Question: "What is the EBITDA for Q3 2023?"

   Vector Search finds:
   ✓ "The Q3 earnings showed strong performance..."
   ✓ "Third quarter results exceeded expectations..."
   ✗ Might miss exact "EBITDA" if not semantically close

   Keyword Search finds:
   ✓ "EBITDA: $45M (Q3 2023)"
   ✗ Might miss "earnings before interest, taxes..."

   Hybrid Search finds BOTH:
   ✓ "EBITDA: $45M (Q3 2023)"
   ✓ "The Q3 earnings showed strong performance..."
   ```

   ```python
   # Hybrid search combines results using RRF
   def hybrid_search(query, top_k=5):
       # Vector search
       vector_results = vector_search(query, top_k=top_k*2)

       # Keyword search (BM25)
       keyword_results = keyword_search(query, top_k=top_k*2)

       # Reciprocal Rank Fusion
       combined = reciprocal_rank_fusion(vector_results, keyword_results)

       return combined[:top_k]
   ```

4. **Query the RAG System**

   ```python
   # Simple query
   result = rag.query("What was the Q3 revenue?")

   print(f"Answer: {result['answer']}")
   print(f"Sources: {result['sources']}")

   # Output:
   # Answer: Based on the Q3 financial report, revenue was $45 million,
   #         representing a 15% increase from Q2. [Source: Q3_Report.pdf]
   # Sources: ['Q3_Report.pdf', 'Earnings_Call_Transcript.docx']
   ```

5. **The Generation Prompt**

   ```python
   SYSTEM_PROMPT = """You are a helpful assistant that answers questions
   based on the provided context. Always cite your sources.

   Rules:
   1. Only use information from the provided context
   2. If the answer is not in the context, say "I don't have enough information"
   3. Cite sources using [Source: filename] format
   4. Be concise but complete"""

   USER_PROMPT = """Context:
   {context}

   Question: {question}

   Answer:"""
   ```

### Testing the RAG System

```python
# Test queries
test_queries = [
    "What was the Q3 revenue?",
    "Who is the CEO?",
    "What technologies does the company use?",
    "When was the last board meeting?",
]

for query in test_queries:
    result = rag.query(query)
    print(f"Q: {query}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {result['sources']}")
    print("-" * 50)
```

---

## Step 4: Measure Baseline Metrics

### What We're Doing
Establishing baseline performance metrics to compare against GraphRAG and ReAct later.

### Why Metrics Matter

Without metrics, you can't know if improvements actually help!

```
Baseline RAG:     MRR=0.65, NDCG=0.70
+ GraphRAG:       MRR=0.75, NDCG=0.80  ← +15% improvement!
+ ReAct Agent:    MRR=0.82, NDCG=0.85  ← +26% improvement!
```

### Key Metrics Explained

| Metric | What It Measures | Range | Intuition |
|--------|------------------|-------|-----------|
| **MRR** | Mean Reciprocal Rank | 0-1 | "How high is the correct answer ranked?" |
| **NDCG** | Normalized Discounted Cumulative Gain | 0-1 | "Are good results at the top?" |
| **Precision@K** | Relevant in top K | 0-1 | "What % of top-K are relevant?" |
| **Recall@K** | Found in top K | 0-1 | "What % of all relevant did we find?" |

### MRR Example

```
Query: "Who is the CEO?"
Correct answer in chunk: chunk_42

Results returned:
  1. chunk_15 (wrong)
  2. chunk_42 (correct!) ← Position 2
  3. chunk_78 (wrong)

Reciprocal Rank = 1/2 = 0.5

Average across all queries = MRR
```

### Instructions

1. **Run the Baseline Metrics Notebook**

   ```
   notebooks/03_vector_index/03_baseline_metrics.py
   ```

2. **Create Test Dataset**

   You need queries with known relevant documents:

   ```python
   test_queries = [
       {
           "query": "What was Q3 revenue?",
           "relevant_doc_ids": ["doc_123", "doc_456"],
           "ground_truth": "Q3 revenue was $45 million"
       },
       {
           "query": "Who is the CEO?",
           "relevant_doc_ids": ["doc_789"],
           "ground_truth": "John Smith is the CEO"
       },
       # ... more test queries
   ]
   ```

3. **Calculate Metrics**

   ```python
   from retrieval.metrics import RetrievalMetrics

   metrics = RetrievalMetrics()

   results = []
   for test in test_queries:
       # Get retrieval results
       retrieved = rag.retrieve(test["query"], top_k=10)
       retrieved_ids = [r.doc_id for r in retrieved]

       # Calculate metrics
       mrr = metrics.calculate_mrr(retrieved_ids, test["relevant_doc_ids"])
       ndcg = metrics.calculate_ndcg(retrieved_ids, test["relevant_doc_ids"], k=10)
       precision = metrics.calculate_precision_at_k(retrieved_ids, test["relevant_doc_ids"], k=5)
       recall = metrics.calculate_recall_at_k(retrieved_ids, test["relevant_doc_ids"], k=10)

       results.append({
           "query": test["query"],
           "mrr": mrr,
           "ndcg": ndcg,
           "precision@5": precision,
           "recall@10": recall
       })

   # Aggregate
   avg_mrr = np.mean([r["mrr"] for r in results])
   avg_ndcg = np.mean([r["ndcg"] for r in results])
   ```

4. **Save Baseline Results**

   ```python
   baseline_results = {
       "system": "baseline_rag",
       "metrics": {
           "mrr": avg_mrr,
           "ndcg@10": avg_ndcg,
           "precision@5": avg_precision,
           "recall@10": avg_recall
       },
       "num_queries": len(test_queries),
       "timestamp": datetime.now().isoformat()
   }

   # Save to Delta Lake
   spark.createDataFrame([baseline_results]).write.format("delta") \
       .save("/mnt/datalake/gold/baseline_metrics")
   ```

### Expected Output

```
BASELINE RAG METRICS
====================
Queries tested: 100

MRR:           0.65
NDCG@10:       0.70
Precision@5:   0.60
Recall@10:     0.75

These are your baseline numbers to beat!
```

---

## Phase 3 Checklist

Before moving to Phase 4, verify:

- [ ] Embeddings generated
  - [ ] All chunks have embeddings
  - [ ] Embedding dimension is 3072
  - [ ] Saved to Gold layer

- [ ] Azure AI Search index created
  - [ ] HNSW algorithm configured
  - [ ] All documents uploaded
  - [ ] Index is queryable

- [ ] RAG pipeline working
  - [ ] Hybrid search functional
  - [ ] Answers include citations
  - [ ] Response quality acceptable

- [ ] Baseline metrics recorded
  - [ ] MRR calculated
  - [ ] NDCG calculated
  - [ ] Results saved for comparison

---

## Verification Queries

```python
# Test search index
from azure.search.documents import SearchClient

client = SearchClient(endpoint, "document-chunks", credential)

# Vector search test
results = client.search(
    search_text=None,
    vector_queries=[{
        "vector": embeddings.embed_query("What is the revenue?"),
        "k_nearest_neighbors": 5,
        "fields": "text_vector"
    }]
)
print(f"Vector search results: {len(list(results))}")

# Hybrid search test
results = client.search(
    search_text="revenue Q3",
    vector_queries=[{
        "vector": embeddings.embed_query("What is the revenue?"),
        "k_nearest_neighbors": 5,
        "fields": "text_vector"
    }]
)
print(f"Hybrid search results: {len(list(results))}")
```

---

## Cost Estimation

| Operation | Volume | Est. Cost |
|-----------|--------|-----------|
| Embeddings (text-embedding-3-large) | 2M chunks | ~$100 |
| Azure AI Search (S1) | 30GB index | ~$250/month |
| RAG Queries (gpt-4o) | Testing | ~$20 |
| **Total Setup** | | **~$120** |
| **Monthly** | | **~$250** |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Search returns no results | Index not populated | Check document upload logs |
| Low similarity scores | Embedding mismatch | Ensure same model for query & docs |
| Slow search | Index not optimized | Increase efSearch parameter |
| Wrong language results | No language filter | Add filter to search query |
| Rate limits on embedding | Too fast | Add delays between batches |

---

## What's Next

In **Phase 4: GraphRAG**, we will:
1. Extract entities and relationships from chunks
2. Build a knowledge graph in Cosmos DB
3. Detect communities of related entities
4. Generate community summaries

This will improve retrieval for complex questions!

---

*Phase 3 Complete! Proceed to [Phase 4: GraphRAG](./PHASE_4_GRAPHRAG.md)*
