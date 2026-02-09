# Project Execution Guide

A simple step-by-step guide to execute each phase of the GraphRAG + ReAct Knowledge Retrieval System.

---

## Phase 1: Data Ingestion (Weeks 1-2)

**Goal:** Load all enterprise data into the system.

### What to do:

1. **Set up Azure environment**
   - Create Azure Databricks workspace
   - Set up ADLS Gen2 storage account
   - Configure Databricks secrets for API keys
   - Run: `notebooks/00_setup/00_azure_environment_setup.py`

2. **Ingest emails from PST files**
   - Upload PST files to Bronze layer in ADLS
   - Extract emails, attachments, and metadata
   - Build email threads (group replies together)
   - Run: `notebooks/01_ingestion/01_pst_email_ingestion.py`

3. **Ingest documents**
   - Upload PDF, DOCX, XLSX, PPTX, MSG files to Bronze layer
   - Extract text content from each document type
   - Store raw documents with metadata
   - Run: `notebooks/01_ingestion/02_document_ingestion.py`

4. **Detect language**
   - Identify English vs Dutch content
   - Tag each document with its language
   - This helps with NER in Phase 2
   - Run: `notebooks/01_ingestion/03_language_detection.py`

### Output:
- Raw data stored in Bronze layer (`/mnt/datalake/bronze/`)
- Documents table with metadata
- Emails table with threading info

---

## Phase 2: NLP Processing (Weeks 3-4)

**Goal:** Process text into clean, structured chunks ready for search.

### What to do:

1. **Chunk documents semantically**
   - Split long documents into smaller pieces
   - Keep related sentences together (semantic chunking)
   - Each chunk is 512-1024 tokens
   - Run: `notebooks/02_nlp_processing/01_semantic_chunking.py`

2. **Extract named entities (NER)**
   - Find people names, organizations, locations, dates
   - Use English model for EN content
   - Use Dutch model for NL content
   - Run: `notebooks/02_nlp_processing/02_ner_extraction.py`

3. **Anonymize PII data**
   - Find sensitive information (SSN, phone, email, IBAN, etc.)
   - Replace with fake but realistic values
   - Supports international formats (US, EU, global)
   - Run: `notebooks/02_nlp_processing/03_anonymization.py`

4. **Generate summaries**
   - Create short summary for each chunk
   - Create summary for each full document
   - Use GPT-4o for high quality summaries
   - Run: `notebooks/02_nlp_processing/04_summarization.py`

### Output:
- Processed chunks in Silver layer (`/mnt/datalake/silver/`)
- Chunks with NER tags
- Anonymized versions of sensitive data
- Summaries for quick understanding

---

## Phase 3: Vector Index & Basic RAG (Week 5)

**Goal:** Create searchable index and basic question-answering system.

### What to do:

1. **Create embeddings and index**
   - Generate vector embeddings for each chunk (text-embedding-3-large)
   - Create Azure AI Search index with HNSW algorithm
   - Enable hybrid search (vector + keyword)
   - Run: `notebooks/03_vector_index/01_embedding_indexing.py`

2. **Build basic RAG pipeline**
   - Connect LangChain to Azure AI Search
   - Set up retrieval chain (find relevant chunks)
   - Set up generation chain (answer questions with GPT-4o)
   - Run: `notebooks/03_vector_index/02_basic_rag.py`

3. **Measure baseline performance**
   - Test with sample questions
   - Calculate MRR (Mean Reciprocal Rank)
   - Calculate NDCG (ranking quality)
   - This is your baseline to beat
   - Run: `notebooks/03_vector_index/03_baseline_metrics.py`

### Output:
- Azure AI Search index with vectors
- Working RAG system for Q&A
- Baseline metrics to compare against

---

## Phase 4: GraphRAG Construction (Weeks 6-8)

**Goal:** Build knowledge graph to understand entity relationships.

### What to do:

1. **Extract entities and relationships**
   - Use GPT-4o to find entities in each chunk
   - Entity types: Person, Organization, Project, Technology, etc.
   - Find relationships: "works on", "reports to", "uses", etc.
   - Run: `notebooks/04_graphrag/01_entity_extraction.py`

2. **Build knowledge graph**
   - Store entities as nodes (vertices)
   - Store relationships as edges
   - Use Cosmos DB Gremlin or in-memory graph
   - Run: `notebooks/04_graphrag/02_knowledge_graph.py`

3. **Detect communities**
   - Group related entities into communities (clusters)
   - Use Leiden algorithm at multiple resolutions
   - Level 0 = small groups, Level 2 = big groups
   - Run: `notebooks/04_graphrag/03_community_detection.py`

4. **Generate community summaries**
   - Write a summary for each community
   - Identify key themes and key entities
   - These summaries help answer "big picture" questions
   - Run: `notebooks/04_graphrag/04_community_summarization.py`

### Output:
- Knowledge graph with entities and relationships
- Hierarchical communities (fine → coarse)
- Community summaries indexed for search

---

## Phase 5: ReAct Agent (Weeks 9-10)

**Goal:** Build intelligent agent that can reason and use tools.

### What to do:

1. **Create combined retriever**
   - Combine vector search + graph search + community search
   - Classify queries as "local" (specific) or "global" (themes)
   - Route queries to best retrieval method
   - Run: `notebooks/05_react_agent/01_graphrag_retriever.py`

2. **Build ReAct agent**
   - Create tools for the agent to use:
     - `vector_search`: Find relevant document chunks
     - `entity_lookup`: Get info about a specific entity
     - `relationship_search`: Find connections
     - `community_search`: Get high-level themes
     - `graph_traversal`: Explore entity neighborhoods
   - Agent thinks, acts, observes, repeats until answer found
   - Run: `notebooks/05_react_agent/02_react_agent.py`

3. **Test multi-hop question answering**
   - Test with complex questions requiring multiple steps
   - Compare: Baseline vs GraphRAG vs ReAct vs Full System
   - Measure which system works best for which question type
   - Run: `notebooks/05_react_agent/03_multi_hop_qa.py`

### Output:
- GraphRAG retriever with query routing
- ReAct agent with 5 tools
- Multi-hop QA capability
- Comparison of all 4 systems

---

## Phase 6: Evaluation (Week 11)

**Goal:** Measure quality and generate final report for thesis.

### What to do:

1. **Run RAGAS evaluation**
   - Measure 4 metrics for each system:
     - **Faithfulness**: Is the answer based on the context?
     - **Answer Relevancy**: Does the answer match the question?
     - **Context Precision**: Is the retrieved context relevant?
     - **Context Recall**: Did we retrieve all needed info?
   - Run: `notebooks/06_evaluation/01_ragas_evaluation.py`

2. **Compare systems statistically**
   - Run paired t-tests between systems
   - Calculate effect sizes (Cohen's d)
   - Find which differences are significant (p < 0.05)
   - Analyze performance by question type
   - Run: `notebooks/06_evaluation/02_comparative_analysis.py`

3. **Generate final report**
   - Create HTML report with charts
   - Create Markdown report for thesis
   - Include:
     - Executive summary
     - Metrics tables
     - Visualizations (bar charts, radar charts, heatmaps)
     - Recommendations
     - Conclusions
   - Run: `notebooks/06_evaluation/03_final_report.py`

### Output:
- RAGAS scores for all systems
- Statistical significance results
- HTML and Markdown evaluation reports
- Visualizations for thesis

---

## Quick Reference: Notebook Execution Order

```
Phase 1:
  1. 00_setup/00_azure_environment_setup.py
  2. 01_ingestion/01_pst_email_ingestion.py
  3. 01_ingestion/02_document_ingestion.py
  4. 01_ingestion/03_language_detection.py

Phase 2:
  5. 02_nlp_processing/01_semantic_chunking.py
  6. 02_nlp_processing/02_ner_extraction.py
  7. 02_nlp_processing/03_anonymization.py
  8. 02_nlp_processing/04_summarization.py

Phase 3:
  9. 03_vector_index/01_embedding_indexing.py
  10. 03_vector_index/02_basic_rag.py
  11. 03_vector_index/03_baseline_metrics.py

Phase 4:
  12. 04_graphrag/01_entity_extraction.py
  13. 04_graphrag/02_knowledge_graph.py
  14. 04_graphrag/03_community_detection.py
  15. 04_graphrag/04_community_summarization.py

Phase 5:
  16. 05_react_agent/01_graphrag_retriever.py
  17. 05_react_agent/02_react_agent.py
  18. 05_react_agent/03_multi_hop_qa.py

Phase 6:
  19. 06_evaluation/01_ragas_evaluation.py
  20. 06_evaluation/02_comparative_analysis.py
  21. 06_evaluation/03_final_report.py
```

---

## Tips for Success

1. **Run notebooks in order** - Each notebook depends on previous ones
2. **Check Delta Lake tables** - Verify data is saved after each step
3. **Monitor costs** - GPT-4o calls cost money, use sampling during testing
4. **Save checkpoints** - Delta Lake has versioning, use it
5. **Test with small data first** - Use `SAMPLE_SIZE` variables before full runs

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| API key errors | Check Databricks secrets are configured |
| Out of memory | Reduce batch size or sample size |
| Slow processing | Use parallel processing where available |
| Missing dependencies | Run `%pip install` cells first |
| Delta table not found | Check previous notebook completed successfully |

---

## Azure Resources Needed

- **Azure Databricks** - Notebook execution
- **Azure OpenAI** - GPT-4o and text-embedding-3-large
- **Azure AI Search** - Vector and hybrid search
- **Azure Cosmos DB** - Graph storage (optional, can use in-memory)
- **ADLS Gen2** - Delta Lake storage

---

*Last updated: February 2026*
*Author: Muhammad Rafiq - KU Leuven Master Thesis*
