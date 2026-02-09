# Phase 5: ReAct Agent

**Duration:** Weeks 9-10
**Goal:** Build intelligent agent that can reason and use tools

---

## Overview

### What We're Building

In this phase, we create an intelligent question-answering system:
1. Combined retriever (vector + graph + community search)
2. Query classifier (local vs global queries)
3. ReAct agent with tool-augmented reasoning
4. Multi-hop question answering capability

### Why This Matters

**Basic RAG Limitation:**
```
Question: "What projects did people who worked with John on Azure also work on?"

Basic RAG:
1. Search for "John" + "Azure" + "projects"
2. Return chunks that contain these words
3. Miss the multi-hop reasoning required
Result: Incomplete or wrong answer

ReAct Agent:
1. THINK: I need to find who worked with John on Azure
2. ACT: entity_lookup("John") → Azure OpenAI project
3. OBSERVE: John worked on Azure OpenAI
4. THINK: Now find John's collaborators
5. ACT: relationship_search("John", "COLLABORATES_WITH")
6. OBSERVE: Sarah, Mike, Lisa collaborate with John
7. THINK: Now find their other projects
8. ACT: entity_lookup("Sarah") → Also works on Copilot
9. OBSERVE: Sarah works on Azure Copilot
10. ANSWER: "Sarah, Mike, and Lisa worked with John on Azure.
            Sarah also worked on Azure Copilot..."
```

### The ReAct Advantage

| Aspect | Basic RAG | GraphRAG | ReAct Agent |
|--------|-----------|----------|-------------|
| Single-hop queries | Good | Good | Good |
| Multi-hop queries | Poor | Medium | Excellent |
| Entity relationships | No | Yes | Yes |
| Reasoning trace | No | No | Yes |
| Tool usage | No | No | Yes |
| Self-correction | No | No | Yes |

---

## Prerequisites

### From Phase 4

- [ ] Knowledge graph built with entities and relationships
- [ ] Communities detected at multiple levels
- [ ] Community summaries generated and indexed

### Azure Resources

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| Azure OpenAI | Agent LLM | GPT-4o deployment |
| Azure AI Search | Vector search | Standard S1 |
| Azure Cosmos DB | Graph queries | Gremlin API |

### Python Dependencies

```python
# Add to requirements.txt
langgraph>=0.1.0        # Agent state machine
langchain>=0.2.0        # Tool framework
langchain-openai>=0.1.0 # OpenAI integration
```

---

## Step 1: Build GraphRAG Retriever

### What We're Doing

Creating a unified retriever that combines all our search capabilities.

### Why

- **Query Routing**: Different queries need different search strategies
- **Unified Interface**: Single API for all retrieval methods
- **Optimal Results**: Use the best method for each query type

### Query Classification

```
LOCAL Queries (specific facts):
├── "Who is John Smith?"
├── "When did Project Alpha start?"
├── "What technology does Team X use?"
└── → Use: Vector search + Entity lookup

GLOBAL Queries (themes/summaries):
├── "What are the main projects?"
├── "Summarize the engineering teams"
├── "What topics does the company focus on?"
└── → Use: Community summaries

HYBRID Queries (both):
├── "How does John's work relate to company strategy?"
├── "What is Project Alpha and how does it fit the bigger picture?"
└── → Use: All methods combined
```

### Instructions

1. **Run the GraphRAG Retriever Notebook**

   ```
   notebooks/05_react_agent/01_graphrag_retriever.py
   ```

2. **Understanding the Query Classifier**

   ```python
   from src.agents.graphrag_retriever import QueryClassifier, QueryType

   classifier = QueryClassifier()

   # Classify incoming queries
   query = "Who worked on Project Alpha?"
   query_type = classifier.classify(query)
   # Returns: QueryType.LOCAL

   query = "What are the main themes in the company?"
   query_type = classifier.classify(query)
   # Returns: QueryType.GLOBAL
   ```

3. **The Classification Prompt**

   ```python
   CLASSIFICATION_PROMPT = """
   Classify this query into one of three types:

   LOCAL: Specific questions about particular entities, facts, or details
   - Questions with specific names, dates, or identifiers
   - "Who", "What", "When" questions about specific things

   GLOBAL: Broad questions about themes, summaries, or overviews
   - Questions asking for summaries or main points
   - "What are the main...", "Summarize...", "Overview of..."

   HYBRID: Questions that need both specific and broad information
   - Questions connecting specific entities to broader themes
   - Multi-part questions

   Query: {query}

   Return only: LOCAL, GLOBAL, or HYBRID
   """
   ```

4. **Building the Combined Retriever**

   ```python
   from src.agents.graphrag_retriever import GraphRAGRetriever

   retriever = GraphRAGRetriever(
       # Vector search configuration
       search_endpoint=search_endpoint,
       search_key=search_key,
       index_name="chunks-index",

       # Graph configuration
       cosmos_endpoint=cosmos_endpoint,
       cosmos_key=cosmos_key,

       # Community summaries index
       community_index_name="community-summaries",

       # Retrieval parameters
       vector_k=10,          # Top 10 vector results
       entity_k=5,           # Top 5 entity results
       community_k=3         # Top 3 community summaries
   )
   ```

5. **How Retrieval Works**

   ```python
   # The retrieve method handles routing automatically

   context = retriever.retrieve(query="Who is John Smith?")

   # Returns GraphRAGContext with:
   # - chunks: Relevant document chunks (from vector search)
   # - entities: Related entities (from graph)
   # - relationships: Entity connections (from graph)
   # - community_summaries: Theme summaries (from community index)
   # - query_type: LOCAL, GLOBAL, or HYBRID

   # For LOCAL query like above:
   # - Mainly uses vector search + entity lookup
   # - Community summaries minimal

   # For GLOBAL query:
   # - Mainly uses community summaries
   # - Vector search + entities minimal
   ```

### Expected Output

```
Retriever Capabilities:
├── Vector Search: Search 100,000 chunks
├── Entity Search: Search 50,000 entities
├── Relationship Search: Query 200,000 edges
├── Community Search: Search 1,050 summaries
└── Query Classification: Auto-route queries

Gold Layer:
├── /mnt/datalake/gold/retriever_config/
│   └── Saved retriever configuration
└── /mnt/datalake/gold/query_logs/
    └── Query classification logs (for analysis)
```

---

## Step 2: Build ReAct Agent

### What We're Doing

Creating an agent that can reason step-by-step and use tools to find answers.

### Why

- **Reasoning**: Agent thinks before acting
- **Tool Use**: Agent can use multiple tools
- **Self-Correction**: Agent can retry if first approach fails
- **Transparency**: See the agent's reasoning trace

### The ReAct Loop

```
┌─────────────────────────────────────────────────────────┐
│                     ReAct Agent                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Question → [THINK] → [ACT] → [OBSERVE] → Loop...       │
│                 │         │         │                    │
│                 ▼         ▼         ▼                    │
│            "I need    Use tool   Get result              │
│             to..."    to search  from tool               │
│                                                          │
│   When confident → [ANSWER]                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Agent Tools

| Tool | Purpose | When Used |
|------|---------|-----------|
| `vector_search` | Find relevant document chunks | Looking for specific text/facts |
| `entity_lookup` | Get information about an entity | Finding entity details |
| `relationship_search` | Find entity connections | Understanding relationships |
| `community_search` | Get theme summaries | Answering broad questions |
| `graph_traversal` | Explore entity neighborhoods | Multi-hop exploration |

### Instructions

1. **Run the ReAct Agent Notebook**

   ```
   notebooks/05_react_agent/02_react_agent.py
   ```

2. **Understanding Agent Tools**

   ```python
   from src.agents.tools import create_agent_tools

   tools = create_agent_tools(
       search_client=search_client,
       graph_client=graph_client,
       community_index=community_index
   )

   # Tools available to the agent:
   # 1. vector_search(query: str, k: int) -> List[chunks]
   # 2. entity_lookup(name: str) -> Entity details
   # 3. relationship_search(entity: str, rel_type: str) -> Relationships
   # 4. community_search(query: str) -> Community summaries
   # 5. graph_traversal(entity: str, depth: int) -> Neighborhood
   ```

3. **Creating the ReAct Agent**

   ```python
   from src.agents.react_agent import ReActAgent

   agent = ReActAgent(
       model="gpt-4o",
       tools=tools,
       max_iterations=10,      # Maximum reasoning steps
       verbose=True            # Show reasoning trace
   )
   ```

4. **How the Agent Reasons**

   ```python
   # Example reasoning trace

   Question: "What projects did John's collaborators work on?"

   Step 1:
   THOUGHT: I need to find who John collaborates with first
   ACTION: entity_lookup("John")
   OBSERVATION: John Smith is a Senior Engineer at Microsoft,
                works on Azure OpenAI project

   Step 2:
   THOUGHT: Now I need to find John's collaborators
   ACTION: relationship_search("John Smith", "COLLABORATES_WITH")
   OBSERVATION: John collaborates with: Sarah Johnson, Mike Lee

   Step 3:
   THOUGHT: Now I need to find what projects Sarah works on
   ACTION: entity_lookup("Sarah Johnson")
   OBSERVATION: Sarah Johnson works on Azure Copilot, Azure OpenAI

   Step 4:
   THOUGHT: Now I need to find what projects Mike works on
   ACTION: entity_lookup("Mike Lee")
   OBSERVATION: Mike Lee works on Azure Kubernetes Service

   Step 5:
   THOUGHT: I now have enough information to answer
   ANSWER: "John's collaborators (Sarah Johnson and Mike Lee)
           work on: Azure Copilot, Azure OpenAI, and
           Azure Kubernetes Service."
   ```

5. **LangGraph State Machine**

   ```python
   # The agent uses LangGraph for state management

   from langgraph.graph import StateGraph

   # States:
   # - "think": Agent decides what to do
   # - "act": Agent uses a tool
   # - "observe": Agent processes tool result
   # - "answer": Agent provides final answer

   # Transitions:
   # think → act (when tool needed)
   # act → observe (after tool execution)
   # observe → think (continue reasoning)
   # think → answer (when confident)
   ```

### Expected Output

```
Agent Capabilities:
├── 5 tools available
├── Max 10 reasoning steps
├── Reasoning trace logging
└── Source attribution

Gold Layer:
├── /mnt/datalake/gold/agent_config/
│   └── Agent configuration and prompts
└── /mnt/datalake/gold/agent_traces/
    └── Reasoning traces for analysis
```

---

## Step 3: Multi-Hop Question Answering

### What We're Doing

Testing the system with complex questions that require multiple reasoning steps.

### Why

- **Thesis Validation**: Prove the system works better than baseline
- **Comparative Analysis**: See which approach works best
- **Question Type Analysis**: Understand system strengths/weaknesses

### Multi-Hop Question Types

```
Type 1: Bridge Questions
"What technology does John's manager use?"
→ Find John → Find manager → Find their technology

Type 2: Comparison Questions
"How do the projects that Sarah and Mike work on differ?"
→ Find Sarah's projects → Find Mike's projects → Compare

Type 3: Aggregation Questions
"How many people work on Azure-related projects?"
→ Find all Azure projects → Find all workers → Count unique

Type 4: Temporal Questions
"What did the team work on before Project Alpha?"
→ Find Project Alpha → Find team → Find earlier projects
```

### Instructions

1. **Run the Multi-Hop QA Notebook**

   ```
   notebooks/05_react_agent/03_multi_hop_qa.py
   ```

2. **Creating Test Questions**

   ```python
   # Multi-hop test questions

   test_questions = [
       # 1-hop (baseline should handle)
       {
           "question": "Who is John Smith?",
           "hops": 1,
           "type": "factual"
       },

       # 2-hop (GraphRAG advantage)
       {
           "question": "What projects does John's manager lead?",
           "hops": 2,
           "type": "bridge"
       },

       # 3-hop (ReAct advantage)
       {
           "question": "What technologies are used by teams that John collaborates with?",
           "hops": 3,
           "type": "aggregation"
       },

       # Global (Community advantage)
       {
           "question": "What are the main research themes in the organization?",
           "hops": 0,
           "type": "global"
       }
   ]
   ```

3. **Running Comparative Tests**

   ```python
   from src.agents.react_agent import MultiHopQAAgent

   # Initialize all systems
   baseline_rag = BasicRAG(...)
   graphrag = GraphRAGRetriever(...)
   react_agent = ReActAgent(...)
   full_system = MultiHopQAAgent(...)  # ReAct + GraphRAG

   # Test each question with each system
   results = []

   for question in test_questions:
       # Run baseline
       baseline_result = baseline_rag.answer(question["question"])

       # Run GraphRAG only
       graphrag_result = graphrag.answer(question["question"])

       # Run ReAct only
       react_result = react_agent.invoke(question["question"])

       # Run full system (ReAct + GraphRAG)
       full_result = full_system.invoke(question["question"])

       results.append({
           "question": question,
           "baseline": baseline_result,
           "graphrag": graphrag_result,
           "react": react_result,
           "full_system": full_result
       })
   ```

4. **The Multi-Hop QA Agent**

   ```python
   from src.agents.react_agent import MultiHopQAAgent

   agent = MultiHopQAAgent(
       model="gpt-4o",
       retriever=graphrag_retriever,
       tools=tools,
       decompose_questions=True  # Break complex questions into sub-questions
   )

   # For complex questions, the agent:
   # 1. Decomposes the question into sub-questions
   # 2. Answers each sub-question
   # 3. Synthesizes final answer

   result = agent.invoke("What projects did John's collaborators work on?")

   # Returns:
   # {
   #   "answer": "...",
   #   "sub_questions": ["Who are John's collaborators?", "What do they work on?"],
   #   "sub_answers": [...],
   #   "reasoning_trace": [...],
   #   "sources": [...],
   #   "tools_used": [...]
   # }
   ```

5. **Measuring Performance**

   ```python
   # Metrics to track for each system

   metrics = {
       "answer_quality": "Human evaluation 1-5",
       "factual_accuracy": "Are facts correct?",
       "completeness": "Did it answer fully?",
       "reasoning_steps": "How many steps needed?",
       "latency": "Time to answer",
       "tool_efficiency": "Tools used vs needed"
   }
   ```

### Expected Output

```
Multi-Hop QA Results:
├── 50 test questions
├── 4 systems compared
├── Metrics for each question/system pair
└── Performance by question type

Gold Layer:
├── /mnt/datalake/gold/qa_test_questions/
│   └── Test question bank
├── /mnt/datalake/gold/qa_results/
│   └── System answers for each question
└── /mnt/datalake/gold/qa_metrics/
    └── Performance metrics
```

### Preliminary Results Pattern

Based on GraphRAG literature, expect:

| Question Type | Baseline | GraphRAG | ReAct | Full System |
|---------------|----------|----------|-------|-------------|
| 1-hop factual | Good | Good | Good | Good |
| 2-hop bridge | Poor | Good | Medium | Best |
| 3-hop complex | Poor | Medium | Good | Best |
| Global themes | Poor | Good | Medium | Good |

---

## Phase 5 Checklist

Before moving to Phase 6, verify:

- [ ] GraphRAG Retriever built
  - [ ] Query classifier working
  - [ ] Vector search integrated
  - [ ] Graph search integrated
  - [ ] Community search integrated

- [ ] ReAct Agent built
  - [ ] All 5 tools working
  - [ ] Reasoning trace logging
  - [ ] Error handling in place
  - [ ] Max iterations set

- [ ] Multi-Hop QA tested
  - [ ] Test questions created
  - [ ] All 4 systems compared
  - [ ] Results saved to Delta Lake
  - [ ] Preliminary analysis done

---

## Verification Queries

```python
# Test query classification
from src.agents.graphrag_retriever import QueryClassifier

classifier = QueryClassifier()
test_queries = [
    "Who is John Smith?",                    # Should be LOCAL
    "What are the main projects?",           # Should be GLOBAL
    "How does John relate to main themes?"   # Should be HYBRID
]

for query in test_queries:
    print(f"{query} → {classifier.classify(query)}")

# Test ReAct agent
from src.agents.react_agent import ReActAgent

agent = ReActAgent(...)
result = agent.invoke("Who are John's collaborators?")
print(f"Answer: {result['answer']}")
print(f"Tools used: {result['tools_used']}")
print(f"Steps: {len(result['reasoning_trace'])}")

# Check QA results
qa_df = spark.read.format("delta").load("/mnt/datalake/gold/qa_results")
qa_df.groupBy("system", "question_type").agg(
    avg("answer_quality").alias("avg_quality")
).show()
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Agent loops forever | No termination condition | Set max_iterations |
| Wrong tool selected | Poor tool descriptions | Improve tool docstrings |
| Query misclassified | Ambiguous query | Use few-shot examples |
| Slow responses | Too many tool calls | Increase retrieval k |
| Missing relationships | Graph query issues | Check Cosmos DB connection |
| Empty community results | Index not populated | Verify community index |

---

## Cost Estimation

```
Multi-Hop QA Testing:
- 50 test questions
- 4 systems
- Avg 5 LLM calls per ReAct response
- Avg 1000 tokens per call

ReAct calls: 50 questions * 5 calls * 1000 tokens = 250,000 tokens
Other systems: 50 * 3 * 1000 = 150,000 tokens
Total tokens: 400,000

Cost at $0.01/1K tokens: ~$4

Full system development/testing: ~$50-100
```

---

## What's Next

In **Phase 6: Evaluation**, we will:
1. Run comprehensive RAGAS evaluation
2. Perform statistical significance testing
3. Analyze performance by question type
4. Generate final thesis report

---

*Phase 5 Complete! Proceed to [Phase 6: Evaluation](./PHASE_6_EVALUATION.md)*
