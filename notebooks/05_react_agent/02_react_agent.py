# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 5.2: ReAct Agent
# MAGIC
# MAGIC Tool-augmented reasoning agent using LangGraph for multi-hop question answering.
# MAGIC
# MAGIC **Week 9-10 - ReAct Agent Implementation**
# MAGIC
# MAGIC ## Features
# MAGIC - ReAct (Reasoning + Acting) loop
# MAGIC - LangGraph state machine
# MAGIC - Multiple specialized tools
# MAGIC - Citation tracking
# MAGIC - Multi-hop reasoning
# MAGIC
# MAGIC ## Author
# MAGIC Muhammad Rafiq - KU Leuven Master Thesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-openai langgraph azure-search-documents numpy delta-spark

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
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append("/Workspace/Repos/mater_thesis/src")

from agents.tools import (
    VectorSearchTool,
    EntityLookupTool,
    RelationshipSearchTool,
    CommunitySummaryTool,
    GraphTraversalTool,
    create_agent_tools
)
from agents.react_agent import (
    ReActAgent,
    ReActConfig,
    create_react_graph
)
from agents.graphrag_retriever import GraphRAGRetriever, GraphRAGConfig
from graphrag.graph_store import InMemoryGraphStore

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

print("Azure credentials loaded")

# COMMAND ----------

# DBTITLE 1,Configure Paths
GOLD_PATH = "/mnt/datalake/gold"

ENTITIES_TABLE = f"{GOLD_PATH}/entities"
RELATIONSHIPS_TABLE = f"{GOLD_PATH}/relationships"
COMMUNITY_SUMMARIES_TABLE = f"{GOLD_PATH}/community_summaries_indexed"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Knowledge Base

# COMMAND ----------

# DBTITLE 1,Load Entities and Relationships
# Load entities
entities_df = spark.read.format("delta").load(ENTITIES_TABLE)
entities = [row.asDict() for row in entities_df.collect()]
print(f"Loaded {len(entities)} entities")

# Load relationships
relationships_df = spark.read.format("delta").load(RELATIONSHIPS_TABLE)
relationships = [row.asDict() for row in relationships_df.collect()]
print(f"Loaded {len(relationships)} relationships")

# Build lookup dictionaries
entity_dict = {e["id"]: e for e in entities}
entity_names = {e["name"].lower(): e["id"] for e in entities}

# COMMAND ----------

# DBTITLE 1,Load Community Summaries
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
print(f"Loaded {len(community_summaries)} community summaries")

# COMMAND ----------

# DBTITLE 1,Initialize Embeddings and Vector Store
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-01"
)

# Initialize vector store
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name="document-chunks",
    embedding_function=embeddings.embed_query,
)

print("Embeddings and vector store initialized")

# COMMAND ----------

# DBTITLE 1,Initialize Graph Store
graph_store = InMemoryGraphStore()

for entity in entities:
    graph_store.add_entity(entity)

for rel in relationships:
    graph_store.add_relationship(rel)

print(f"Graph store: {graph_store.get_statistics()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Agent Tools

# COMMAND ----------

# DBTITLE 1,Initialize Tools
# Create agent tools
tools = create_agent_tools(
    vector_store=vector_store,
    graph_store=graph_store,
    entities=entities,
    relationships=relationships,
    community_summaries=community_summaries,
    community_embeddings=community_embeddings,
    embeddings=embeddings
)

print(f"Created {len(tools)} agent tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")

# COMMAND ----------

# DBTITLE 1,Test Individual Tools
# Test vector search tool
print("Testing vector_search tool:")
print("-" * 40)
vector_tool = next(t for t in tools if t.name == "vector_search")
result = vector_tool.invoke({"query": "project management", "top_k": 3})
print(result[:500])

# COMMAND ----------

# DBTITLE 1,Test Entity Lookup Tool
# Test entity lookup
print("\nTesting entity_lookup tool:")
print("-" * 40)
entity_tool = next(t for t in tools if t.name == "entity_lookup")

if entities:
    sample_name = entities[0]["name"]
    result = entity_tool.invoke({"entity_name": sample_name, "include_relationships": True})
    print(result[:500])

# COMMAND ----------

# DBTITLE 1,Test Community Search Tool
# Test community search
print("\nTesting community_search tool:")
print("-" * 40)
community_tool = next(t for t in tools if t.name == "community_search")
result = community_tool.invoke({"query": "technology and projects", "level": None})
print(result[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize ReAct Agent

# COMMAND ----------

# DBTITLE 1,Configure LLM
# Initialize LLM for agent
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_deployment="gpt-4o",
    temperature=0.1,
    max_tokens=2000,
)

print("LLM initialized")

# COMMAND ----------

# DBTITLE 1,Configure ReAct Agent
# Agent configuration
agent_config = ReActConfig(
    model_deployment="gpt-4o",
    temperature=0.1,
    max_tokens=2000,
    max_reasoning_steps=6,
    require_citations=True,
    max_tool_calls_per_step=3,
    include_reasoning_trace=True  # Enable for debugging
)

print("Agent configuration:")
print(f"  Max reasoning steps: {agent_config.max_reasoning_steps}")
print(f"  Max tool calls per step: {agent_config.max_tool_calls_per_step}")
print(f"  Include reasoning trace: {agent_config.include_reasoning_trace}")

# COMMAND ----------

# DBTITLE 1,Initialize ReAct Agent
# Create ReAct agent
agent = ReActAgent(
    llm=llm,
    tools=tools,
    config=agent_config
)

print("ReAct agent initialized with tools:")
for tool_name in agent.tool_map.keys():
    print(f"  - {tool_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test ReAct Agent

# COMMAND ----------

# DBTITLE 1,Test Simple Query
# Test with a simple query
simple_query = "What is the main purpose of the organization?"

print(f"Query: {simple_query}")
print("=" * 60)

result = agent.invoke(simple_query)

print(f"\n📝 ANSWER:\n{result['answer']}")

print(f"\n🔧 TOOLS USED: {result['tools_used']}")

print(f"\n📚 SOURCES CITED:")
for source in result['sources'][:5]:
    print(f"  - [{source['type']}] {source['source']}")

print(f"\n📊 REASONING STEPS: {result['steps']}")

# COMMAND ----------

# DBTITLE 1,Test Entity-Focused Query
# Test entity-focused query
if entities:
    entity_name = entities[0]["name"]
    entity_query = f"Tell me about {entity_name} and their role in the organization."

    print(f"Query: {entity_query}")
    print("=" * 60)

    result = agent.invoke(entity_query)

    print(f"\n📝 ANSWER:\n{result['answer']}")
    print(f"\n🔧 TOOLS USED: {result['tools_used']}")
    print(f"\n📊 REASONING STEPS: {result['steps']}")

# COMMAND ----------

# DBTITLE 1,Test Relationship Query
# Test relationship query
relationship_query = "What are the key relationships between different teams in the organization?"

print(f"Query: {relationship_query}")
print("=" * 60)

result = agent.invoke(relationship_query)

print(f"\n📝 ANSWER:\n{result['answer']}")
print(f"\n🔧 TOOLS USED: {result['tools_used']}")

# Show reasoning trace
if result.get('reasoning_trace'):
    print(f"\n🧠 REASONING TRACE:")
    for step in result['reasoning_trace']:
        print(f"  {step}")

# COMMAND ----------

# DBTITLE 1,Test Complex Query
# Test complex multi-hop query
complex_query = "What technologies are used by the projects that involve the engineering team, and how do these relate to recent company initiatives?"

print(f"Query: {complex_query}")
print("=" * 60)

result = agent.invoke(complex_query)

print(f"\n📝 ANSWER:\n{result['answer']}")
print(f"\n🔧 TOOLS USED: {result['tools_used']}")
print(f"\n📚 SOURCES CITED: {len(result['sources'])} sources")
print(f"\n📊 REASONING STEPS: {result['steps']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LangGraph Agent (Alternative)

# COMMAND ----------

# DBTITLE 1,Create LangGraph Agent
# Create LangGraph-based agent
langgraph_agent = create_react_graph(
    llm=llm,
    tools=tools,
    config=agent_config
)

print("LangGraph agent created")

# COMMAND ----------

# DBTITLE 1,Test LangGraph Agent
from langchain_core.messages import HumanMessage

# Test LangGraph agent
test_query = "What are the main projects and who leads them?"

initial_state = {
    "messages": [HumanMessage(content=test_query)],
    "current_step": 0,
    "max_steps": agent_config.max_reasoning_steps,
    "tools_used": [],
    "sources_cited": [],
    "final_answer": None,
    "reasoning_trace": []
}

print(f"Query: {test_query}")
print("=" * 60)

# Run LangGraph
final_state = langgraph_agent.invoke(initial_state)

# Extract answer
if final_state.get("final_answer"):
    print(f"\n📝 ANSWER:\n{final_state['final_answer']}")
else:
    # Get last AI message
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            print(f"\n📝 ANSWER:\n{msg.content}")
            break

print(f"\n🔧 TOOLS USED: {final_state.get('tools_used', [])}")
print(f"\n📊 STEPS: {final_state.get('current_step', 0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Agent Performance Analysis

# COMMAND ----------

# DBTITLE 1,Batch Query Testing
# Test multiple queries
test_queries = [
    "Who are the key people in the organization?",
    "What projects are currently active?",
    "What technologies does the company use?",
    "How are different departments connected?",
    "What are the main themes in recent communications?"
]

results_summary = []

for query in test_queries:
    print(f"Processing: {query[:50]}...")

    try:
        result = agent.invoke(query)
        results_summary.append({
            "query": query,
            "answer_length": len(result['answer']),
            "tools_used": len(result['tools_used']),
            "sources_cited": len(result['sources']),
            "steps": result['steps'],
            "success": True
        })
    except Exception as e:
        results_summary.append({
            "query": query,
            "answer_length": 0,
            "tools_used": 0,
            "sources_cited": 0,
            "steps": 0,
            "success": False
        })

# Display results
results_df = spark.createDataFrame(results_summary)
display(results_df)

# COMMAND ----------

# DBTITLE 1,Tool Usage Analysis
from collections import Counter

# Analyze tool usage across all queries
all_tools_used = []
for r in results_summary:
    if "tools_used" in r:
        all_tools_used.extend([str(r["tools_used"])])

# Aggregate from agent results
print("Tool Usage Analysis:")
print("-" * 40)

# Count tool invocations
tool_counts = {}
for result in results_summary:
    # Note: This is simplified - in production, track actual tool names
    print(f"  Query answered in {result['steps']} steps using {result['tools_used']} tool calls")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Agent Configuration

# COMMAND ----------

# DBTITLE 1,Save Configuration
# Save agent configuration
agent_config_data = {
    "model_deployment": agent_config.model_deployment,
    "temperature": agent_config.temperature,
    "max_tokens": agent_config.max_tokens,
    "max_reasoning_steps": agent_config.max_reasoning_steps,
    "require_citations": agent_config.require_citations,
    "max_tool_calls_per_step": agent_config.max_tool_calls_per_step,
    "tools_available": [t.name for t in tools],
    "created_at": datetime.now().isoformat()
}

agent_config_df = spark.createDataFrame([agent_config_data])
agent_config_df.write.format("delta").mode("overwrite").save(f"{GOLD_PATH}/agent_config")

print(f"Saved agent configuration to {GOLD_PATH}/agent_config")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary

# COMMAND ----------

# DBTITLE 1,Phase 5.2 Summary
print("=" * 60)
print("PHASE 5.2: REACT AGENT COMPLETE")
print("=" * 60)

print(f"\n🤖 AGENT CAPABILITIES:")
print(f"  - ReAct reasoning loop (think → act → observe)")
print(f"  - {len(tools)} specialized tools")
print(f"  - Citation tracking")
print(f"  - Multi-hop reasoning (up to {agent_config.max_reasoning_steps} steps)")

print(f"\n🔧 AVAILABLE TOOLS:")
for tool in tools:
    print(f"  - {tool.name}")

print(f"\n📊 TEST RESULTS:")
successful = sum(1 for r in results_summary if r['success'])
print(f"  Queries tested: {len(results_summary)}")
print(f"  Successful: {successful}/{len(results_summary)}")
avg_steps = sum(r['steps'] for r in results_summary) / len(results_summary) if results_summary else 0
print(f"  Avg reasoning steps: {avg_steps:.1f}")

print(f"\n📁 OUTPUT:")
print(f"  Configuration: {GOLD_PATH}/agent_config")

print("\n" + "=" * 60)
print("NEXT: 03_multi_hop_qa.py - Multi-Hop Question Answering")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **ReAct Loop:**
# MAGIC 1. THINK - Analyze what information is needed
# MAGIC 2. ACT - Use appropriate tools
# MAGIC 3. OBSERVE - Review tool results
# MAGIC 4. REPEAT - Continue until answer is found
# MAGIC 5. ANSWER - Provide response with citations
# MAGIC
# MAGIC **Tool Selection:**
# MAGIC - `vector_search` - Specific facts and evidence
# MAGIC - `entity_lookup` - Entity details
# MAGIC - `relationship_search` - Connections between entities
# MAGIC - `community_search` - High-level themes
# MAGIC - `graph_traversal` - Entity neighborhoods
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Test multi-hop question answering
# MAGIC 2. Evaluate against baseline RAG
# MAGIC 3. Fine-tune tool selection
