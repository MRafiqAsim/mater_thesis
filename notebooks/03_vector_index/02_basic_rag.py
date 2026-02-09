# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Basic RAG Implementation
# MAGIC
# MAGIC **Phase 3: Vector Index & Basic RAG | Week 5**
# MAGIC
# MAGIC This notebook implements the baseline RAG chain using Azure AI Search and GPT-4o.
# MAGIC
# MAGIC ## Objectives
# MAGIC - Implement LangChain RAG chain with Azure AI Search retriever
# MAGIC - Configure hybrid search (vector + keyword + semantic ranking)
# MAGIC - Build answer generation with source citations
# MAGIC - Create interactive Q&A interface
# MAGIC
# MAGIC ## RAG Pipeline
# MAGIC ```
# MAGIC Query → Embedding → Hybrid Search → Context Assembly → GPT-4o → Answer with Citations
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install langchain langchain-openai azure-search-documents tiktoken

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Azure credentials
OPENAI_ENDPOINT = dbutils.secrets.get("azure-openai", "endpoint")
OPENAI_KEY = dbutils.secrets.get("azure-openai", "api-key")
SEARCH_ENDPOINT = dbutils.secrets.get("azure-search", "endpoint")
SEARCH_KEY = dbutils.secrets.get("azure-search", "api-key")

# RAG Configuration
RAG_CONFIG = {
    "index_name": "knowledge-chunks",
    "top_k": 5,
    "search_type": "hybrid",  # vector, keyword, hybrid
    "use_semantic_ranker": True,
    "model_deployment": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 1000,
}

print(f"Index: {RAG_CONFIG['index_name']}")
print(f"Search Type: {RAG_CONFIG['search_type']}")
print(f"Model: {RAG_CONFIG['model_deployment']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Search & Embedding Clients

# COMMAND ----------

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import time

# Initialize search client
search_client = SearchClient(
    SEARCH_ENDPOINT,
    RAG_CONFIG["index_name"],
    AzureKeyCredential(SEARCH_KEY)
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
)

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment=RAG_CONFIG["model_deployment"],
    temperature=RAG_CONFIG["temperature"],
    max_tokens=RAG_CONFIG["max_tokens"],
)

print("All clients initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define RAG Components

# COMMAND ----------

class HybridRetriever:
    """Hybrid retriever using Azure AI Search."""

    def __init__(self, search_client, embeddings, config):
        self.search_client = search_client
        self.embeddings = embeddings
        self.config = config

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search."""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Build vector query
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self.config["top_k"],
            fields="content_vector"
        )

        # Execute search based on type
        search_kwargs = {
            "top": self.config["top_k"],
            "select": ["chunk_id", "content", "summary", "parent_document_id",
                      "parent_type", "source_file", "entities"],
        }

        if self.config["search_type"] == "vector":
            search_kwargs["search_text"] = None
            search_kwargs["vector_queries"] = [vector_query]

        elif self.config["search_type"] == "keyword":
            search_kwargs["search_text"] = query

        else:  # hybrid
            search_kwargs["search_text"] = query
            search_kwargs["vector_queries"] = [vector_query]

            if self.config.get("use_semantic_ranker"):
                search_kwargs["query_type"] = "semantic"
                search_kwargs["semantic_configuration_name"] = "semantic-config"

        # Execute search
        results = self.search_client.search(**search_kwargs)

        # Process results
        documents = []
        for result in results:
            documents.append({
                "chunk_id": result["chunk_id"],
                "content": result.get("content", ""),
                "summary": result.get("summary", ""),
                "score": result["@search.score"],
                "reranker_score": result.get("@search.reranker_score"),
                "source_file": result.get("source_file", "Unknown"),
                "parent_type": result.get("parent_type", "document"),
                "entities": result.get("entities", []),
            })

        retrieval_time = (time.time() - start_time) * 1000

        return documents, retrieval_time

# Initialize retriever
retriever = HybridRetriever(search_client, embeddings, RAG_CONFIG)
print("Retriever initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define RAG Prompts

# COMMAND ----------

# System prompt for RAG
SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context from enterprise documents and emails.

Guidelines:
- Answer ONLY based on the information in the provided context
- If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
- Cite your sources using [1], [2], etc. format corresponding to the context numbers
- Be concise but comprehensive
- Maintain factual accuracy - do not make up information
- If multiple sources support a point, cite all relevant ones
- Preserve important names, dates, and specific details from the sources"""

# RAG prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Remember to cite sources using [1], [2], etc."""),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Build RAG Chain

# COMMAND ----------

class RAGChain:
    """Complete RAG chain with retrieval and generation."""

    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.chain = prompt | llm | StrOutputParser()

    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents as context."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.get("source_file", "Unknown")
            doc_type = doc.get("parent_type", "document")
            content = doc.get("content", "")
            summary = doc.get("summary", "")

            # Include summary if available
            if summary:
                context_parts.append(
                    f"[{i}] Source: {source} ({doc_type})\nSummary: {summary}\nContent: {content}"
                )
            else:
                context_parts.append(
                    f"[{i}] Source: {source} ({doc_type})\n{content}"
                )

        return "\n\n---\n\n".join(context_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """Execute RAG query."""
        # Retrieve
        documents, retrieval_time = self.retriever.retrieve(question)

        # Format context
        context = self.format_context(documents)

        # Generate
        generation_start = time.time()
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        generation_time = (time.time() - generation_start) * 1000

        # Build response
        return {
            "answer": answer,
            "sources": [
                {
                    "index": i + 1,
                    "chunk_id": doc["chunk_id"],
                    "source_file": doc["source_file"],
                    "score": doc["score"],
                    "reranker_score": doc.get("reranker_score"),
                    "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                }
                for i, doc in enumerate(documents)
            ],
            "context": context,
            "metrics": {
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "total_time_ms": retrieval_time + generation_time,
                "num_sources": len(documents),
            }
        }

# Initialize RAG chain
rag_chain = RAGChain(retriever, llm, RAG_PROMPT)
print("RAG chain initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test RAG Chain

# COMMAND ----------

def display_rag_response(response: Dict):
    """Display RAG response in a formatted way."""
    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(response["answer"])

    print("\n" + "=" * 70)
    print("SOURCES:")
    print("=" * 70)
    for source in response["sources"]:
        score_str = f"Score: {source['score']:.4f}"
        if source.get("reranker_score"):
            score_str += f", Reranker: {source['reranker_score']:.4f}"
        print(f"\n[{source['index']}] {source['source_file']} ({score_str})")
        print(f"    {source['preview']}")

    print("\n" + "=" * 70)
    print("METRICS:")
    print("=" * 70)
    m = response["metrics"]
    print(f"  Retrieval Time: {m['retrieval_time_ms']:.1f} ms")
    print(f"  Generation Time: {m['generation_time_ms']:.1f} ms")
    print(f"  Total Time: {m['total_time_ms']:.1f} ms")
    print(f"  Sources Used: {m['num_sources']}")

# Test query
test_question = "What are the main objectives of the project?"
response = rag_chain.query(test_question)
display_rag_response(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Interactive Q&A

# COMMAND ----------

# Test multiple queries
test_queries = [
    "Who are the key stakeholders involved?",
    "What decisions were made regarding the project timeline?",
    "What technologies are being used in this project?",
    "Are there any risks or concerns mentioned?",
]

for query in test_queries:
    print(f"\n{'#' * 70}")
    print(f"QUESTION: {query}")
    print('#' * 70)

    response = rag_chain.query(query)
    print(f"\nANSWER:\n{response['answer']}")
    print(f"\nTime: {response['metrics']['total_time_ms']:.1f}ms, Sources: {response['metrics']['num_sources']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare Search Types

# COMMAND ----------

def compare_search_types(query: str):
    """Compare vector, keyword, and hybrid search."""
    print(f"\nQuery: {query}\n")
    print("-" * 80)

    results = {}

    for search_type in ["vector", "keyword", "hybrid"]:
        # Update config
        RAG_CONFIG["search_type"] = search_type

        # Create new retriever with updated config
        temp_retriever = HybridRetriever(search_client, embeddings, RAG_CONFIG)
        temp_rag = RAGChain(temp_retriever, llm, RAG_PROMPT)

        # Query
        response = temp_rag.query(query)

        results[search_type] = {
            "answer": response["answer"][:300] + "...",
            "time_ms": response["metrics"]["total_time_ms"],
            "top_source": response["sources"][0]["source_file"] if response["sources"] else "N/A",
            "top_score": response["sources"][0]["score"] if response["sources"] else 0,
        }

        print(f"\n{search_type.upper()} SEARCH:")
        print(f"  Time: {results[search_type]['time_ms']:.1f}ms")
        print(f"  Top Source: {results[search_type]['top_source']} (score: {results[search_type]['top_score']:.4f})")
        print(f"  Answer Preview: {results[search_type]['answer'][:200]}...")

    # Reset to hybrid
    RAG_CONFIG["search_type"] = "hybrid"

    return results

# Compare search types
comparison_query = "What is the project budget and timeline?"
compare_search_types(comparison_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save RAG Configuration

# COMMAND ----------

import json

STORAGE_ACCOUNT = dbutils.secrets.get("azure-storage", "account-name")
CONTAINER = dbutils.secrets.get("azure-storage", "container-name")
BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_PATH = f"{BASE_PATH}/gold"

# Save RAG configuration
rag_config_export = {
    "index_name": RAG_CONFIG["index_name"],
    "top_k": RAG_CONFIG["top_k"],
    "search_type": RAG_CONFIG["search_type"],
    "use_semantic_ranker": RAG_CONFIG["use_semantic_ranker"],
    "model_deployment": RAG_CONFIG["model_deployment"],
    "temperature": RAG_CONFIG["temperature"],
    "system_prompt": SYSTEM_PROMPT,
}

config_path = f"{GOLD_PATH}/rag_config/config.json"
dbutils.fs.put(config_path, json.dumps(rag_config_export, indent=2), overwrite=True)
print(f"Saved RAG configuration to {config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║              BASIC RAG IMPLEMENTATION COMPLETE                   ║
╠══════════════════════════════════════════════════════════════════╣
║  RAG COMPONENTS:                                                 ║
║  • Retriever: Azure AI Search (Hybrid)                           ║
║  • Embeddings: text-embedding-3-large (3072 dims)                ║
║  • Generator: GPT-4o                                             ║
║  • Ranking: Semantic Reranker                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  CONFIGURATION:                                                  ║
║  • Index: {RAG_CONFIG['index_name']:<52} ║
║  • Top K: {RAG_CONFIG['top_k']:<53} ║
║  • Search Type: {RAG_CONFIG['search_type']:<46} ║
║  • Temperature: {RAG_CONFIG['temperature']:<46} ║
╠══════════════════════════════════════════════════════════════════╣
║  CAPABILITIES:                                                   ║
║  • Vector search (semantic similarity)                           ║
║  • Keyword search (BM25)                                         ║
║  • Hybrid search (combined)                                      ║
║  • Source citations [1], [2], etc.                               ║
║  • Latency tracking                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEPS:                                                     ║
║  1. Run 03_baseline_metrics.py to measure MRR, NDCG              ║
║  2. Record baseline for comparison with GraphRAG/ReAct           ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
