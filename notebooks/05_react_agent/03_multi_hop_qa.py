# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 5.3: Multi-Hop Question Answering
# MAGIC
# MAGIC Advanced multi-hop reasoning with question decomposition and answer synthesis.
# MAGIC
# MAGIC **Week 10 - Multi-Hop QA & Evaluation**
# MAGIC
# MAGIC ## Features
# MAGIC - Question decomposition into sub-questions
# MAGIC - Independent sub-question answering
# MAGIC - Answer synthesis with source aggregation
# MAGIC - Comparison: Baseline RAG vs GraphRAG vs ReAct
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai langgraph azure-search-documents numpy pandas delta-spark ragas

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from agents.tools import create_agent_tools
from agents.react_agent import ReActAgent, MultiHopQAAgent, ReActConfig
from agents.graphrag_retriever import GraphRAGRetriever, GraphRAGConfig
from graphrag.graph_store import InMemoryGraphStore
from retrieval.rag_chain import RAGChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Load Azure Configuration
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="azure-openai", key="endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="azure-openai", key="api-key")
AZURE_SEARCH_ENDPOINT = dbutils.secrets.get(scope="azure-search", key="endpoint")
AZURE_SEARCH_KEY = dbutils.secrets.get(scope="azure-search", key="admin-key")

print("Credentials loaded")

# COMMAND ----------

# DBTITLE 1,Configure Paths
GOLD_PATH = "/mnt/datalake/gold"

ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"
COMMUNITY_SUMMARIES_TABLE = f"{GOLD_PATH}/community_summaries_indexed"

# Output
EVALUATION_TABLE = f"{GOLD_PATH}/qa_evaluation_results"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Components

# COMMAND ----------

# DBTITLE 1,Load Knowledge Base
# Load entities
entities_df = spark.read.format("delta").load(ENTITIES_TABLE)
entities = [row.asDict() for row in entities_df.collect()]
print(f"Entities: {len(entities)}")

# Load relationships
relationships_df = spark.read.format("delta").load(RELATIONSHIPS_TABLE)
relationships = [row.asDict() for row in relationships_df.collect()]
print(f"Relationships: {len(relationships)}")

# Load community summaries
summaries_df = spark.read.format("delta").load(COMMUNITY_SUMMARIES_TABLE)
summaries_raw = summaries_df.collect()

community_summaries = []
community_embeddings = []

for row in summaries_raw:
    summary = row.asDict()
    if "key_entities" in summary and isinstance(summary["key_entities"], str):
        summary["key_entities"] = json.loads(summary["key_entities"])
    if "key_themes" in summary and isinstance(summary["key_themes"], str):
        summary["key_themes"] = json.loads(summary["key_themes"])
    embedding = json.loads(summary.pop("embedding", "[]"))
    community_summaries.append(summary)
    community_embeddings.append(embedding)

community_embeddings = np.array(community_embeddings)
print(f"Community summaries: {len(community_summaries)}")

# COMMAND ----------

# DBTITLE 1,Initialize LLM and Embeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-01"
)

# Vector store
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name="document-chunks",
    embedding_function=embeddings.embed_query,
)

# LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="gpt-4o",
    temperature=0.1,
)

# Graph store
graph_store = InMemoryGraphStore()
for entity in entities:
    graph_store.add_entity(entity)
for rel in relationships:
    graph_store.add_relationship(rel)

print("Components initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize QA Systems

# COMMAND ----------

# DBTITLE 1,1. Baseline RAG (Vector Search Only)
# Simple RAG chain - vector search only
baseline_rag = RAGChain(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    search_endpoint=AZURE_SEARCH_ENDPOINT,
    search_key=AZURE_SEARCH_KEY,
    index_name="document-chunks"
)

print("1. Baseline RAG initialized (vector search only)")

# COMMAND ----------

# DBTITLE 1,2. GraphRAG Retriever
# GraphRAG - combines vector + graph + communities
graphrag_retriever = GraphRAGRetriever(
    vector_store=vector_store,
    graph_store=graph_store,
    community_summaries=community_summaries,
    community_embeddings=community_embeddings,
    embeddings=embeddings,
    entities=entities,
    relationships=relationships,
    config=GraphRAGConfig(
        vector_top_k=5,
        entity_top_k=5,
        community_top_k=3
    )
)

print("2. GraphRAG retriever initialized")

# COMMAND ----------

# DBTITLE 1,3. ReAct Agent
# Create tools
tools = create_agent_tools(
    vector_store=vector_store,
    graph_store=graph_store,
    entities=entities,
    relationships=relationships,
    community_summaries=community_summaries,
    community_embeddings=community_embeddings,
    embeddings=embeddings
)

# ReAct agent configuration
react_config = ReActConfig(
    max_reasoning_steps=6,
    max_tool_calls_per_step=3,
    include_reasoning_trace=True
)

# ReAct agent
react_agent = ReActAgent(llm=llm, tools=tools, config=react_config)

print(f"3. ReAct agent initialized with {len(tools)} tools")

# COMMAND ----------

# DBTITLE 1,4. Multi-Hop QA Agent
# Multi-hop QA agent
multihop_agent = MultiHopQAAgent(llm=llm, tools=tools, config=react_config)

print("4. Multi-Hop QA agent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Test Questions

# COMMAND ----------

# DBTITLE 1,Create Test Question Set
# Multi-hop test questions requiring multiple reasoning steps
test_questions = [
    # Simple single-hop questions
    {
        "question": "What is the main focus of the engineering team?",
        "type": "single-hop",
        "expected_hops": 1
    },
    {
        "question": "Who are the key people in the organization?",
        "type": "single-hop",
        "expected_hops": 1
    },

    # Two-hop questions
    {
        "question": "What projects involve the people who work in engineering?",
        "type": "two-hop",
        "expected_hops": 2
    },
    {
        "question": "Which technologies are used by the projects that the sales team supports?",
        "type": "two-hop",
        "expected_hops": 2
    },

    # Multi-hop questions (3+)
    {
        "question": "How do the technologies used by engineering projects relate to the company's strategic initiatives mentioned in recent communications?",
        "type": "multi-hop",
        "expected_hops": 3
    },
    {
        "question": "What are the connections between key people, their projects, and the technologies they use, and how does this align with organizational goals?",
        "type": "multi-hop",
        "expected_hops": 4
    },

    # Global/summary questions
    {
        "question": "Provide an overview of the company's main activities and how different teams collaborate.",
        "type": "global",
        "expected_hops": 2
    },
    {
        "question": "What are the main themes and patterns across all organizational communications?",
        "type": "global",
        "expected_hops": 2
    }
]

print(f"Created {len(test_questions)} test questions")
for q in test_questions[:3]:
    print(f"  - [{q['type']}] {q['question'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Multi-Hop QA Evaluation

# COMMAND ----------

# DBTITLE 1,Evaluate Baseline RAG
print("=" * 60)
print("EVALUATING: Baseline RAG (Vector Search Only)")
print("=" * 60)

baseline_results = []

for q in test_questions:
    print(f"\nQ: {q['question'][:60]}...")

    try:
        # Baseline RAG
        result = baseline_rag.query(q['question'])

        baseline_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": result.get('answer', '')[:500],
            "sources": len(result.get('sources', [])),
            "success": True
        })
        print(f"  ✓ Answer length: {len(result.get('answer', ''))}")

    except Exception as e:
        baseline_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": f"Error: {str(e)}",
            "sources": 0,
            "success": False
        })
        print(f"  ✗ Error: {str(e)}")

# COMMAND ----------

# DBTITLE 1,Evaluate GraphRAG + Simple RAG
print("=" * 60)
print("EVALUATING: GraphRAG + RAG")
print("=" * 60)

graphrag_results = []

for q in test_questions:
    print(f"\nQ: {q['question'][:60]}...")

    try:
        # Get GraphRAG context
        context = graphrag_retriever.retrieve(q['question'])
        formatted_context = context.format_context(max_tokens=4000)

        # Use LLM with GraphRAG context
        prompt = f"""Based on the following context, answer the question.

Context:
{formatted_context}

Question: {q['question']}

Answer:"""

        response = llm.invoke(prompt)

        graphrag_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": response.content[:500],
            "sources": len(context.get_all_results()),
            "query_type": context.query_type.value,
            "success": True
        })
        print(f"  ✓ Query type: {context.query_type.value}, Sources: {len(context.get_all_results())}")

    except Exception as e:
        graphrag_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": f"Error: {str(e)}",
            "sources": 0,
            "query_type": "error",
            "success": False
        })
        print(f"  ✗ Error: {str(e)}")

# COMMAND ----------

# DBTITLE 1,Evaluate ReAct Agent
print("=" * 60)
print("EVALUATING: ReAct Agent")
print("=" * 60)

react_results = []

for q in test_questions:
    print(f"\nQ: {q['question'][:60]}...")

    try:
        result = react_agent.invoke(q['question'])

        react_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": result['answer'][:500],
            "sources": len(result.get('sources', [])),
            "tools_used": result.get('tools_used', []),
            "reasoning_steps": result.get('steps', 0),
            "success": True
        })
        print(f"  ✓ Steps: {result.get('steps', 0)}, Tools: {len(result.get('tools_used', []))}")

    except Exception as e:
        react_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": f"Error: {str(e)}",
            "sources": 0,
            "tools_used": [],
            "reasoning_steps": 0,
            "success": False
        })
        print(f"  ✗ Error: {str(e)}")

# COMMAND ----------

# DBTITLE 1,Evaluate Multi-Hop Agent
print("=" * 60)
print("EVALUATING: Multi-Hop QA Agent")
print("=" * 60)

multihop_results = []

for q in test_questions:
    print(f"\nQ: {q['question'][:60]}...")

    try:
        result = multihop_agent.invoke(q['question'])

        multihop_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": result['answer'][:500],
            "sources": len(result.get('sources', [])),
            "sub_questions": len(result.get('sub_answers', [])),
            "actual_hops": result.get('num_hops', 1),
            "expected_hops": q['expected_hops'],
            "success": True
        })
        print(f"  ✓ Hops: {result.get('num_hops', 1)}, Sub-questions: {len(result.get('sub_answers', []))}")

    except Exception as e:
        multihop_results.append({
            "question": q['question'],
            "question_type": q['type'],
            "answer": f"Error: {str(e)}",
            "sources": 0,
            "sub_questions": 0,
            "actual_hops": 0,
            "expected_hops": q['expected_hops'],
            "success": False
        })
        print(f"  ✗ Error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Results

# COMMAND ----------

# DBTITLE 1,Create Comparison Summary
# Build comparison dataframe
comparison_data = []

for i, q in enumerate(test_questions):
    comparison_data.append({
        "question": q['question'][:80] + "...",
        "type": q['type'],
        "baseline_success": baseline_results[i]['success'],
        "baseline_sources": baseline_results[i]['sources'],
        "graphrag_success": graphrag_results[i]['success'],
        "graphrag_sources": graphrag_results[i]['sources'],
        "react_success": react_results[i]['success'],
        "react_steps": react_results[i].get('reasoning_steps', 0),
        "multihop_success": multihop_results[i]['success'],
        "multihop_hops": multihop_results[i].get('actual_hops', 0)
    })

comparison_df = spark.createDataFrame(comparison_data)
display(comparison_df)

# COMMAND ----------

# DBTITLE 1,Aggregate Statistics
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

# Success rates
baseline_success = sum(1 for r in baseline_results if r['success']) / len(baseline_results)
graphrag_success = sum(1 for r in graphrag_results if r['success']) / len(graphrag_results)
react_success = sum(1 for r in react_results if r['success']) / len(react_results)
multihop_success = sum(1 for r in multihop_results if r['success']) / len(multihop_results)

print(f"\n📊 SUCCESS RATES:")
print(f"  Baseline RAG:     {baseline_success*100:.1f}%")
print(f"  GraphRAG:         {graphrag_success*100:.1f}%")
print(f"  ReAct Agent:      {react_success*100:.1f}%")
print(f"  Multi-Hop Agent:  {multihop_success*100:.1f}%")

# Average sources
print(f"\n📚 AVG SOURCES USED:")
print(f"  Baseline RAG:     {np.mean([r['sources'] for r in baseline_results]):.1f}")
print(f"  GraphRAG:         {np.mean([r['sources'] for r in graphrag_results]):.1f}")
print(f"  ReAct Agent:      {np.mean([r['sources'] for r in react_results]):.1f}")
print(f"  Multi-Hop Agent:  {np.mean([r['sources'] for r in multihop_results]):.1f}")

# ReAct specific
print(f"\n🔧 REACT AGENT METRICS:")
avg_steps = np.mean([r.get('reasoning_steps', 0) for r in react_results])
print(f"  Avg reasoning steps: {avg_steps:.1f}")

# Multi-hop specific
print(f"\n🔗 MULTI-HOP AGENT METRICS:")
avg_hops = np.mean([r.get('actual_hops', 0) for r in multihop_results])
avg_subq = np.mean([r.get('sub_questions', 0) for r in multihop_results])
print(f"  Avg actual hops: {avg_hops:.1f}")
print(f"  Avg sub-questions: {avg_subq:.1f}")

# COMMAND ----------

# DBTITLE 1,Compare by Question Type
print("\n📈 PERFORMANCE BY QUESTION TYPE:")
print("-" * 60)

for qtype in ["single-hop", "two-hop", "multi-hop", "global"]:
    type_indices = [i for i, q in enumerate(test_questions) if q['type'] == qtype]

    if type_indices:
        baseline_type = sum(1 for i in type_indices if baseline_results[i]['success']) / len(type_indices)
        graphrag_type = sum(1 for i in type_indices if graphrag_results[i]['success']) / len(type_indices)
        react_type = sum(1 for i in type_indices if react_results[i]['success']) / len(type_indices)
        multihop_type = sum(1 for i in type_indices if multihop_results[i]['success']) / len(type_indices)

        print(f"\n{qtype.upper()} ({len(type_indices)} questions):")
        print(f"  Baseline:  {baseline_type*100:.0f}%  |  GraphRAG: {graphrag_type*100:.0f}%  |  ReAct: {react_type*100:.0f}%  |  MultiHop: {multihop_type*100:.0f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sample Answer Comparison

# COMMAND ----------

# DBTITLE 1,Compare Answers for Multi-Hop Question
# Pick a multi-hop question for detailed comparison
multihop_idx = next(i for i, q in enumerate(test_questions) if q['type'] == "multi-hop")

question = test_questions[multihop_idx]['question']

print(f"QUESTION: {question}")
print("=" * 60)

print("\n📗 BASELINE RAG ANSWER:")
print("-" * 40)
print(baseline_results[multihop_idx]['answer'])

print("\n📘 GRAPHRAG ANSWER:")
print("-" * 40)
print(graphrag_results[multihop_idx]['answer'])

print("\n📙 REACT AGENT ANSWER:")
print("-" * 40)
print(react_results[multihop_idx]['answer'])

print("\n📕 MULTI-HOP AGENT ANSWER:")
print("-" * 40)
print(multihop_results[multihop_idx]['answer'])

# Show sub-questions if available
if multihop_results[multihop_idx].get('sub_questions', 0) > 0:
    print("\n🔗 SUB-QUESTIONS GENERATED:")
    # Note: Full sub-answers would be in the agent result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Evaluation Results

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
# Combine all results
all_results = []

for i, q in enumerate(test_questions):
    all_results.append({
        "question": q['question'],
        "question_type": q['type'],
        "expected_hops": q['expected_hops'],
        # Baseline
        "baseline_answer": baseline_results[i]['answer'],
        "baseline_success": baseline_results[i]['success'],
        "baseline_sources": baseline_results[i]['sources'],
        # GraphRAG
        "graphrag_answer": graphrag_results[i]['answer'],
        "graphrag_success": graphrag_results[i]['success'],
        "graphrag_sources": graphrag_results[i]['sources'],
        # ReAct
        "react_answer": react_results[i]['answer'],
        "react_success": react_results[i]['success'],
        "react_sources": react_results[i]['sources'],
        "react_steps": react_results[i].get('reasoning_steps', 0),
        # Multi-hop
        "multihop_answer": multihop_results[i]['answer'],
        "multihop_success": multihop_results[i]['success'],
        "multihop_sources": multihop_results[i]['sources'],
        "multihop_hops": multihop_results[i].get('actual_hops', 0),
        # Meta
        "evaluated_at": datetime.now().isoformat()
    })

results_df = spark.createDataFrame(all_results)
results_df.write.format("delta").mode("overwrite").save(EVALUATION_TABLE)

print(f"Saved evaluation results to {EVALUATION_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary

# COMMAND ----------

# DBTITLE 1,Phase 5.3 Summary
print("=" * 60)
print("PHASE 5.3: MULTI-HOP QA EVALUATION COMPLETE")
print("=" * 60)

print(f"\n📊 SYSTEMS EVALUATED:")
print(f"  1. Baseline RAG (vector search only)")
print(f"  2. GraphRAG (vector + graph + communities)")
print(f"  3. ReAct Agent (tool-augmented reasoning)")
print(f"  4. Multi-Hop Agent (question decomposition)")

print(f"\n📈 KEY FINDINGS:")
print(f"  - Questions tested: {len(test_questions)}")
print(f"  - Best for single-hop: {'TBD based on results'}")
print(f"  - Best for multi-hop: {'TBD based on results'}")
print(f"  - Most sources used: {'TBD based on results'}")

print(f"\n📁 OUTPUT:")
print(f"  Evaluation results: {EVALUATION_TABLE}")

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE!")
print("NEXT: Phase 6 - Full System Evaluation (Week 11)")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **System Comparison:**
# MAGIC
# MAGIC | System | Strengths | Weaknesses |
# MAGIC |--------|-----------|------------|
# MAGIC | Baseline RAG | Fast, simple | No reasoning, limited context |
# MAGIC | GraphRAG | Rich context, themes | Still single-pass |
# MAGIC | ReAct | Iterative reasoning | Higher latency |
# MAGIC | Multi-Hop | Complex queries | Highest latency |
# MAGIC
# MAGIC **Recommendations:**
# MAGIC - Use Baseline for simple factual queries
# MAGIC - Use GraphRAG for theme/summary questions
# MAGIC - Use ReAct for relationship exploration
# MAGIC - Use Multi-Hop for complex analytical questions
# MAGIC
# MAGIC **Next Phase:**
# MAGIC - P6: Full system evaluation with RAGAS
# MAGIC - Metrics: MRR, NDCG, Faithfulness, Relevance
# MAGIC - User study preparation
