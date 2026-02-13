# Enhanced Architecture: PathRAG + ReAct-Assisted Preprocessing

This document describes the enhanced pipeline architecture that integrates:
1. **ReAct-Assisted Preprocessing** - Agentic reasoning for summarization and anonymization
2. **PathRAG** - Path-based retrieval for better multi-hop reasoning

---

## 1. Architecture Overview

```mermaid
flowchart TB
    subgraph Bronze["🥉 BRONZE LAYER"]
        B1["Raw PST Files"]
        B2["Raw Documents"]
        B3["Raw Attachments"]
    end

    subgraph Silver["🥈 SILVER LAYER + ReAct"]
        S1["Semantic Chunking"]
        S2["Basic NER (spaCy)"]
        S3["Confidence Scoring"]

        subgraph ReActPreprocess["🤖 ReAct Preprocessing"]
            RA1["ReAct Summarization"]
            RA2["ReAct Anonymization"]
            RA3["ReAct Entity Validation"]
        end

        S4["Enriched Chunks"]
    end

    subgraph Gold["🥇 GOLD LAYER + PathRAG"]
        G1["Entities & Relationships"]
        G2["Knowledge Graph"]
        G3["Community Detection"]

        subgraph PathRAGIndex["🛤️ PathRAG Index"]
            PR1["Path Finding"]
            PR2["Path Scoring"]
            PR3["Path Pruning"]
        end

        G4["Embeddings + Vectors"]
    end

    subgraph Retrieval["🔍 RETRIEVAL LAYER"]
        R1["Baseline RAG"]
        R2["GraphRAG"]
        R3["PathRAG"]
        R4["PathRAG + ReAct Agent"]
    end

    subgraph UI["💬 USER INTERFACE"]
        U1["Query Input"]
        U2["Answer + Sources"]
        U3["Reasoning Trace"]
    end

    Bronze --> Silver
    S1 --> S2 --> S3
    S3 -->|"Low Confidence"| ReActPreprocess
    S3 -->|"High Confidence"| S4
    ReActPreprocess --> S4

    Silver --> Gold
    G1 --> G2 --> G3
    G2 --> PathRAGIndex
    G4 --> Retrieval
    PathRAGIndex --> Retrieval

    Retrieval --> UI

    style Bronze fill:#CD7F32,stroke:#8B4513,color:#fff
    style Silver fill:#C0C0C0,stroke:#808080,color:#000
    style Gold fill:#FFD700,stroke:#B8860B,color:#000
    style ReActPreprocess fill:#fff2cc,stroke:#d6b656
    style PathRAGIndex fill:#dae8fc,stroke:#6c8ebf
```

---

## 2. ReAct-Assisted Silver Layer

### 2.1 Processing Flow

```mermaid
flowchart TB
    subgraph Stage1["STAGE 1: Fast Processing (All Chunks)"]
        A1["Input Chunks"]
        A2["Semantic Chunking"]
        A3["Basic NER (spaCy)"]
        A4["Basic PII (Presidio)"]
        A5["Confidence Scoring"]
    end

    subgraph Decision["Confidence Check"]
        D1{{"Confidence ≥ 0.85?"}}
    end

    subgraph HighConf["HIGH CONFIDENCE (95%)"]
        H1["Pass Through"]
        H2["Standard Processing"]
    end

    subgraph LowConf["LOW CONFIDENCE (5%)"]
        subgraph Stage2["STAGE 2: ReAct Reasoning"]
            R1["🤖 ReAct Agent"]
            R2["THINK: Analyze ambiguity"]
            R3["ACT: Use tools"]
            R4["OBSERVE: Get results"]
            R5["OUTPUT: Refined data"]
        end
    end

    subgraph Output["Final Output"]
        O1["Enriched Chunks"]
        O2["Validated Entities"]
        O3["Quality Summaries"]
    end

    A1 --> A2 --> A3 --> A4 --> A5
    A5 --> D1
    D1 -->|"Yes"| HighConf
    D1 -->|"No"| LowConf
    HighConf --> Output
    R1 --> R2 --> R3 --> R4 --> R5
    LowConf --> Output

    style Stage2 fill:#fff2cc,stroke:#d6b656
    style Decision fill:#f8cecc,stroke:#b85450
```

### 2.2 ReAct Preprocessing Tools

| Tool | Purpose | When Used |
|------|---------|-----------|
| `search_document_context` | Search surrounding text for entity clarification | Ambiguous entity names |
| `verify_entity_type` | Verify if entity is PERSON, ORG, or other | Unclear entity types |
| `check_pii_context` | Determine if text is real PII or false positive | Uncertain PII matches |
| `cross_reference_entities` | Check if two mentions refer to same entity | Entity resolution |
| `validate_relationship` | Verify if extracted relationship is valid | Relationship extraction |

### 2.3 Tool Definitions

```python
PREPROCESSING_TOOLS = [
    {
        "name": "search_document_context",
        "description": "Search surrounding text within the document for entity clarification",
        "parameters": {
            "entity": "The entity text to search context for",
            "window_size": "Number of sentences to include (default: 5)"
        }
    },
    {
        "name": "verify_entity_type",
        "description": "Verify if an entity is PERSON, ORGANIZATION, PRODUCT, or other type",
        "parameters": {
            "entity": "The entity text to verify",
            "candidate_types": "List of possible entity types"
        }
    },
    {
        "name": "check_pii_context",
        "description": "Determine if detected text is real PII or a false positive",
        "parameters": {
            "text": "The potential PII text",
            "pii_type": "Type of PII (PERSON, EMAIL, PHONE, etc.)",
            "context": "Surrounding text for analysis"
        }
    },
    {
        "name": "cross_reference_entities",
        "description": "Check if two entity mentions refer to the same real-world entity",
        "parameters": {
            "entity1": "First entity mention",
            "entity2": "Second entity mention",
            "context1": "Context of first mention",
            "context2": "Context of second mention"
        }
    },
    {
        "name": "validate_relationship",
        "description": "Verify if an extracted relationship between entities is valid",
        "parameters": {
            "source_entity": "Source entity of relationship",
            "relationship_type": "Type of relationship",
            "target_entity": "Target entity of relationship",
            "evidence": "Text evidence for the relationship"
        }
    }
]
```

---

## 3. ReAct-Assisted Summarization

### 3.1 Three-Tier Summarization Strategy

```mermaid
flowchart TB
    subgraph Input["Input Classification"]
        I1["All Chunks"]
        I2{{"Complexity?"}}
    end

    subgraph Simple["SIMPLE (60%)"]
        S1["Single Topic"]
        S2["Single LLM Call"]
        S3["'Summarize this chunk'"]
    end

    subgraph Medium["MEDIUM (30%)"]
        M1["2-3 Topics"]
        M2["Structured Prompt"]
        M3["Extract: decisions,<br/>actions, people"]
    end

    subgraph Complex["COMPLEX (10%)"]
        C1["Multi-hop Content"]
        C2["🤖 ReAct Agent"]
        C3["Iterative Reasoning"]
    end

    subgraph Output["Unified Output"]
        O1["Structured Summary"]
        O2["Key Decisions"]
        O3["Stakeholders"]
        O4["Action Items"]
    end

    I1 --> I2
    I2 -->|"1 topic"| Simple
    I2 -->|"2-3 topics"| Medium
    I2 -->|"Multi-hop"| Complex

    Simple --> Output
    Medium --> Output
    Complex --> Output

    style Simple fill:#d5e8d4,stroke:#82b366
    style Medium fill:#dae8fc,stroke:#6c8ebf
    style Complex fill:#fff2cc,stroke:#d6b656
```

### 3.2 ReAct Summarization Example

```
INPUT: Complex email thread (15 messages, 5 participants, multiple topics)

┌─────────────────────────────────────────────────────────────────────┐
│ THINK: This is a long thread with multiple participants.           │
│        I need to identify the main topics first.                   │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: extract_topics(thread)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE: Topics identified:                                        │
│   1. Budget approval request                                       │
│   2. Timeline concerns                                             │
│   3. Resource allocation                                           │
│   4. Vendor selection                                              │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: Multiple decisions might have been made. Let me find them.  │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: find_decisions(thread)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE: Decisions found:                                          │
│   - APPROVED: Budget of $50,000                                    │
│   - APPROVED: Vendor A selected                                    │
│   - PENDING: Timeline needs stakeholder review                     │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: Who are the key decision makers and stakeholders?           │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: identify_stakeholders(thread)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE: Stakeholders:                                             │
│   - John Smith (Budget Approver)                                   │
│   - Sarah Johnson (Project Lead, Requester)                        │
│   - Mike Lee (Technical Lead)                                      │
│   - Lisa Chen (Procurement)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: Now I have enough information to create a structured        │
│        summary covering all key aspects.                           │
├─────────────────────────────────────────────────────────────────────┤
│ OUTPUT:                                                            │
│                                                                    │
│ SUMMARY:                                                           │
│ Email thread discussing Q3 project budget and vendor selection.    │
│ Budget of $50,000 approved by John Smith. Vendor A selected after  │
│ technical review by Mike Lee. Timeline pending final review.       │
│                                                                    │
│ DECISIONS:                                                         │
│ - Budget: $50,000 approved                                         │
│ - Vendor: Vendor A selected                                        │
│                                                                    │
│ PENDING:                                                           │
│ - Timeline finalization (awaiting stakeholder input)               │
│                                                                    │
│ STAKEHOLDERS:                                                      │
│ - John Smith (Approver)                                            │
│ - Sarah Johnson (Lead)                                             │
│ - Mike Lee (Technical)                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Summarization Tools

```python
SUMMARIZATION_TOOLS = [
    {
        "name": "extract_topics",
        "description": "Extract main topics/themes from a document or thread",
        "parameters": {
            "text": "The text to analyze",
            "max_topics": "Maximum number of topics to extract (default: 5)"
        }
    },
    {
        "name": "find_decisions",
        "description": "Identify decisions made in the text (approved, rejected, pending)",
        "parameters": {
            "text": "The text to analyze"
        }
    },
    {
        "name": "identify_stakeholders",
        "description": "Identify key people and their roles in the discussion",
        "parameters": {
            "text": "The text to analyze"
        }
    },
    {
        "name": "extract_action_items",
        "description": "Extract action items and their assignees",
        "parameters": {
            "text": "The text to analyze"
        }
    },
    {
        "name": "get_thread_timeline",
        "description": "Extract chronological sequence of events/messages",
        "parameters": {
            "text": "The email thread or document"
        }
    }
]
```

---

## 4. ReAct-Assisted Anonymization

### 4.1 Smart PII Detection Pipeline

```mermaid
flowchart TB
    subgraph Input["Input Text"]
        I1["Raw Text with Potential PII"]
    end

    subgraph Presidio["STAGE 1: Presidio Scan"]
        P1["Pattern-Based Detection"]
        P2["Entity Recognition"]
        P3["Confidence Scoring"]
    end

    subgraph Classification["Confidence Classification"]
        C1{{"Score Level?"}}

        subgraph High["CERTAIN PII (Score > 0.9)"]
            H1["Auto Anonymize"]
        end

        subgraph Medium["UNCERTAIN PII (0.5 - 0.9)"]
            M1["🤖 ReAct Validation"]
        end

        subgraph Low["LIKELY FALSE POSITIVE (< 0.5)"]
            L1["Keep As-Is"]
        end
    end

    subgraph ReActValidation["ReAct Agent Validation"]
        R1["THINK: Is this real PII?"]
        R2["ACT: Check context"]
        R3["OBSERVE: Evidence"]
        R4["DECIDE: Anonymize or Keep"]
    end

    subgraph Output["Final Output"]
        O1["Anonymized Text"]
        O2["PII Mapping Table"]
        O3["Validation Report"]
    end

    Input --> Presidio
    P1 --> P2 --> P3 --> C1
    C1 -->|"> 0.9"| High
    C1 -->|"0.5 - 0.9"| Medium
    C1 -->|"< 0.5"| Low

    High --> Output
    Medium --> ReActValidation
    ReActValidation --> Output
    Low --> Output

    style ReActValidation fill:#fff2cc,stroke:#d6b656
    style High fill:#f8cecc,stroke:#b85450
    style Low fill:#d5e8d4,stroke:#82b366
```

### 4.2 ReAct Anonymization Example

```
INPUT: "Contact John Deere for the tractor parts. John Smith from HR will process the order."

┌─────────────────────────────────────────────────────────────────────┐
│ PRESIDIO DETECTION:                                                │
│   - "John Deere" → PERSON (confidence: 0.7)                        │
│   - "John Smith" → PERSON (confidence: 0.95)                       │
│   - "HR" → ORGANIZATION (confidence: 0.6)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ AUTO-ANONYMIZE (confidence > 0.9):                                 │
│   - "John Smith" → [PERSON_1]                                      │
│                                                                    │
│ REACT VALIDATION (confidence 0.5-0.9):                             │
│                                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: "John Deere" detected as PERSON with 0.7 confidence.        │
│        This could be a person name or the tractor company.         │
│        I need to check the context.                                │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: check_pii_context("John Deere", "PERSON", context)            │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE: Context contains "tractor parts" - John Deere is a        │
│          well-known tractor/equipment manufacturer.                │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: This is the company name, not a person. Should NOT          │
│        anonymize as it's not PII.                                  │
├─────────────────────────────────────────────────────────────────────┤
│ DECIDE: Keep "John Deere" (company name, not PII)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ THINK: "HR" detected as ORGANIZATION with 0.6 confidence.          │
│        This is a department reference, not a specific org.         │
├─────────────────────────────────────────────────────────────────────┤
│ DECIDE: Keep "HR" (generic department, not identifying)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ FINAL OUTPUT:                                                      │
│ "Contact John Deere for the tractor parts. [PERSON_1] from HR      │
│  will process the order."                                          │
│                                                                    │
│ PII MAPPING:                                                       │
│   [PERSON_1] = "John Smith" (anonymized - real person)             │
│                                                                    │
│ KEPT AS-IS:                                                        │
│   "John Deere" (company name, not PII)                             │
│   "HR" (generic department reference)                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Anonymization Tools

```python
ANONYMIZATION_TOOLS = [
    {
        "name": "check_pii_context",
        "description": "Analyze context to determine if detected text is real PII",
        "parameters": {
            "text": "The potential PII text",
            "pii_type": "Detected PII type (PERSON, EMAIL, PHONE, etc.)",
            "context": "Surrounding text (2-3 sentences)"
        }
    },
    {
        "name": "lookup_known_entities",
        "description": "Check if text matches known non-PII entities (companies, products)",
        "parameters": {
            "text": "Text to look up",
            "entity_type": "Expected entity type"
        }
    },
    {
        "name": "analyze_name_pattern",
        "description": "Analyze if a name follows person name patterns vs other patterns",
        "parameters": {
            "name": "The name to analyze"
        }
    },
    {
        "name": "check_email_signature",
        "description": "Check if text appears in email signature format",
        "parameters": {
            "text": "Text to check",
            "email_content": "Full email content"
        }
    }
]
```

---

## 5. PathRAG Integration

### 5.1 PathRAG vs GraphRAG

```mermaid
flowchart LR
    subgraph Query["User Query"]
        Q1["'How is John connected<br/>to Azure Copilot?'"]
    end

    subgraph GraphRAG["GraphRAG Response"]
        G1["Entities:"]
        G2["• John Smith"]
        G3["• Sarah Johnson"]
        G4["• Azure Copilot"]
        G5["Relationships:"]
        G6["• COLLABORATES_WITH"]
        G7["• WORKS_ON"]
        G8["(LLM must infer connection)"]
    end

    subgraph PathRAG["PathRAG Response"]
        P1["Explicit Path:"]
        P2["John Smith"]
        P3["↓ COLLABORATES_WITH"]
        P4["Sarah Johnson"]
        P5["↓ WORKS_ON"]
        P6["Azure Copilot"]
        P7["(Connection is explicit)"]
    end

    Query --> GraphRAG
    Query --> PathRAG

    style GraphRAG fill:#dae8fc,stroke:#6c8ebf
    style PathRAG fill:#d5e8d4,stroke:#82b366
```

### 5.2 PathRAG Architecture

```mermaid
flowchart TB
    subgraph Input["Query Processing"]
        I1["User Query"]
        I2["Entity Extraction"]
        I3["Query Entities"]
    end

    subgraph PathFinding["Path Finding"]
        PF1["Knowledge Graph"]
        PF2["BFS/DFS Traversal"]
        PF3["Candidate Paths"]
    end

    subgraph PathScoring["Path Scoring"]
        PS1["Relevance to Query"]
        PS2["Path Length Penalty"]
        PS3["Edge Weight Scores"]
        PS4["Scored Paths"]
    end

    subgraph PathPruning["Path Pruning"]
        PP1["Top-K Selection"]
        PP2["Redundancy Removal"]
        PP3["Final Paths"]
    end

    subgraph Context["Context Building"]
        C1["Path Descriptions"]
        C2["Entity Details"]
        C3["Supporting Chunks"]
        C4["PathRAG Context"]
    end

    Input --> PathFinding
    I1 --> I2 --> I3
    I3 --> PF1
    PF1 --> PF2 --> PF3

    PathFinding --> PathScoring
    PF3 --> PS1 --> PS2 --> PS3 --> PS4

    PathScoring --> PathPruning
    PS4 --> PP1 --> PP2 --> PP3

    PathPruning --> Context
    PP3 --> C1 --> C2 --> C3 --> C4

    style PathFinding fill:#e1d5e7,stroke:#9673a6
    style PathScoring fill:#fff2cc,stroke:#d6b656
    style PathPruning fill:#f8cecc,stroke:#b85450
```

### 5.3 Path Data Structure

```python
@dataclass
class ReasoningPath:
    """A path through the knowledge graph"""

    path_id: str
    nodes: List[Entity]          # Ordered list of entities in path
    edges: List[Relationship]    # Relationships connecting nodes
    score: float                 # Relevance score
    length: int                  # Number of hops

    def to_description(self) -> str:
        """Convert path to natural language description"""
        parts = []
        for i, (node, edge) in enumerate(zip(self.nodes[:-1], self.edges)):
            parts.append(f"{node.name} --{edge.type}--> ")
        parts.append(self.nodes[-1].name)
        return "".join(parts)


@dataclass
class PathRAGContext:
    """Context returned by PathRAG retriever"""

    paths: List[ReasoningPath]              # Relevant reasoning paths
    entities: List[Entity]                   # Unique entities from paths
    relationships: List[Relationship]        # Unique relationships from paths
    supporting_chunks: List[Chunk]           # Text chunks supporting paths

    def to_prompt_context(self) -> str:
        """Format for LLM prompt"""
        context = "## Reasoning Paths\n"
        for path in self.paths:
            context += f"- {path.to_description()}\n"

        context += "\n## Entities\n"
        for entity in self.entities:
            context += f"- {entity.name} ({entity.type}): {entity.description}\n"

        context += "\n## Supporting Evidence\n"
        for chunk in self.supporting_chunks:
            context += f"- {chunk.text[:200]}...\n"

        return context
```

### 5.4 PathRAG Retriever Implementation

```python
class PathRAGRetriever:
    """PathRAG retriever with path finding and pruning"""

    def __init__(
        self,
        graph_client: CosmosDBGremlinClient,
        embedding_model: AzureOpenAIEmbeddings,
        max_path_length: int = 4,
        top_k_paths: int = 10
    ):
        self.graph = graph_client
        self.embeddings = embedding_model
        self.max_path_length = max_path_length
        self.top_k_paths = top_k_paths

    def retrieve(self, query: str) -> PathRAGContext:
        """Main retrieval method"""

        # Step 1: Extract entities from query
        query_entities = self._extract_query_entities(query)

        # Step 2: Find candidate paths
        candidate_paths = self._find_paths(query_entities)

        # Step 3: Score paths by relevance
        scored_paths = self._score_paths(candidate_paths, query)

        # Step 4: Prune to top-k
        pruned_paths = self._prune_paths(scored_paths)

        # Step 5: Get supporting chunks
        chunks = self._get_supporting_chunks(pruned_paths, query)

        # Step 6: Build context
        return PathRAGContext(
            paths=pruned_paths,
            entities=self._extract_unique_entities(pruned_paths),
            relationships=self._extract_unique_relationships(pruned_paths),
            supporting_chunks=chunks
        )

    def _find_paths(
        self,
        query_entities: List[str]
    ) -> List[ReasoningPath]:
        """Find all paths between query-relevant entities"""

        paths = []

        # Find paths between all pairs of query entities
        for i, entity1 in enumerate(query_entities):
            for entity2 in query_entities[i+1:]:
                # Gremlin query for paths
                gremlin_query = f"""
                g.V().has('name', '{entity1}')
                 .repeat(both().simplePath())
                 .until(has('name', '{entity2}').or().loops().is(gte({self.max_path_length})))
                 .has('name', '{entity2}')
                 .path()
                 .limit(50)
                """

                result_paths = self.graph.execute(gremlin_query)
                paths.extend(self._parse_paths(result_paths))

        # Also find paths from query entities to important hub nodes
        hub_entities = self._get_hub_entities()
        for entity in query_entities:
            for hub in hub_entities:
                paths.extend(self._find_path_pair(entity, hub))

        return paths

    def _score_paths(
        self,
        paths: List[ReasoningPath],
        query: str
    ) -> List[ReasoningPath]:
        """Score paths by relevance to query"""

        query_embedding = self.embeddings.embed_query(query)

        for path in paths:
            # Component 1: Semantic similarity
            path_text = path.to_description()
            path_embedding = self.embeddings.embed_query(path_text)
            semantic_score = cosine_similarity(query_embedding, path_embedding)

            # Component 2: Path length penalty (shorter = better)
            length_penalty = 1.0 / (1.0 + 0.2 * path.length)

            # Component 3: Edge importance
            edge_score = np.mean([e.weight for e in path.edges])

            # Combined score
            path.score = (
                0.5 * semantic_score +
                0.3 * length_penalty +
                0.2 * edge_score
            )

        return sorted(paths, key=lambda p: p.score, reverse=True)

    def _prune_paths(
        self,
        paths: List[ReasoningPath]
    ) -> List[ReasoningPath]:
        """Prune to top-k non-redundant paths"""

        pruned = []
        seen_entity_pairs = set()

        for path in paths:
            # Create signature for redundancy check
            pair = (path.nodes[0].id, path.nodes[-1].id)

            # Skip if we already have a path for this pair
            if pair in seen_entity_pairs:
                continue

            seen_entity_pairs.add(pair)
            pruned.append(path)

            if len(pruned) >= self.top_k_paths:
                break

        return pruned
```

---

## 6. Retrieval Benchmark Design

### 6.1 Four Systems Comparison

```mermaid
flowchart TB
    subgraph Systems["RETRIEVAL SYSTEMS"]
        subgraph S1["System 1: BASELINE RAG"]
            B1["Query"]
            B2["Vector Search"]
            B3["Top-K Chunks"]
            B4["LLM"]
            B5["Answer"]
            B1 --> B2 --> B3 --> B4 --> B5
        end

        subgraph S2["System 2: GRAPHRAG"]
            G1["Query"]
            G2["Vector Search"]
            G3["Entity Lookup"]
            G4["Community Search"]
            G5["Combined Context"]
            G6["LLM"]
            G7["Answer"]
            G1 --> G2 & G3 & G4 --> G5 --> G6 --> G7
        end

        subgraph S3["System 3: PATHRAG"]
            P1["Query"]
            P2["Vector Search"]
            P3["Path Finding"]
            P4["Path Pruning"]
            P5["Path Context"]
            P6["LLM"]
            P7["Answer"]
            P1 --> P2 & P3 --> P4 --> P5 --> P6 --> P7
        end

        subgraph S4["System 4: PATHRAG + REACT"]
            R1["Query"]
            R2["ReAct Agent"]
            R3["THINK"]
            R4["ACT (PathRAG Tools)"]
            R5["OBSERVE"]
            R6["Multi-step Reasoning"]
            R7["Answer + Trace"]
            R1 --> R2 --> R3 --> R4 --> R5 --> R3
            R3 --> R6 --> R7
        end
    end

    style S1 fill:#f5f5f5,stroke:#666
    style S2 fill:#dae8fc,stroke:#6c8ebf
    style S3 fill:#d5e8d4,stroke:#82b366
    style S4 fill:#fff2cc,stroke:#d6b656
```

### 6.2 Evaluation Metrics

| Metric | Description | Target | Measurement |
|--------|-------------|--------|-------------|
| **Faithfulness** | Answer grounded in retrieved context | >0.85 | RAGAS |
| **Answer Relevancy** | Answer addresses the query | >0.80 | RAGAS |
| **Context Precision** | Retrieved context is relevant | >0.75 | RAGAS |
| **Context Recall** | All needed info was retrieved | >0.80 | RAGAS |
| **Multi-hop Accuracy** | Correct on 2+ hop questions | >0.70 | Custom |
| **Path Validity** | Reasoning paths are correct | >0.85 | Manual |
| **Latency** | Response time | <5s | Measured |
| **Cost per Query** | API cost | <$0.10 | Calculated |

### 6.3 Test Question Categories

```python
TEST_QUESTIONS = {
    "single_hop": [
        # Direct factual questions
        "Who is John Smith?",
        "What project does Sarah lead?",
        "When was the Azure OpenAI project started?",
    ],

    "two_hop": [
        # Requires one intermediate step
        "Who are John's collaborators?",
        "What projects does Microsoft's AI team work on?",
        "Who approved the Q3 budget?",
    ],

    "multi_hop": [
        # Requires 3+ reasoning steps
        "What projects did John's collaborators work on?",
        "Who are the stakeholders connected to Azure Copilot through Sarah?",
        "What decisions were made by people who worked with John?",
    ],

    "global": [
        # Requires community-level understanding
        "What are the main themes in the AI division?",
        "Summarize the organizational structure",
        "What are the key projects across all teams?",
    ]
}
```

### 6.4 Expected Results Matrix

| Question Type | Baseline | GraphRAG | PathRAG | PathRAG+ReAct |
|---------------|----------|----------|---------|---------------|
| Single-hop | 0.85 | 0.88 | 0.88 | 0.90 |
| Two-hop | 0.60 | 0.75 | 0.82 | 0.88 |
| Multi-hop | 0.35 | 0.55 | 0.70 | 0.85 |
| Global | 0.40 | 0.80 | 0.75 | 0.82 |
| **Average** | **0.55** | **0.75** | **0.79** | **0.86** |

---

## 7. ReAct Agent with PathRAG Tools

### 7.1 Agent Architecture

```mermaid
flowchart TB
    subgraph Agent["🤖 ReAct Agent with PathRAG"]
        THINK["💭 THINK<br/>Analyze question<br/>Plan next step"]
        ACT["⚡ ACT<br/>Use PathRAG tool"]
        OBSERVE["👁️ OBSERVE<br/>Process path results"]
        ANSWER["✅ ANSWER<br/>Final response"]

        THINK --> ACT
        ACT --> OBSERVE
        OBSERVE --> THINK
        THINK --> ANSWER
    end

    subgraph Tools["PathRAG Agent Tools"]
        T1["🔢 vector_search<br/>Find similar chunks"]
        T2["👤 entity_lookup<br/>Get entity details"]
        T3["🛤️ find_paths<br/>Find reasoning paths"]
        T4["🔗 traverse_path<br/>Follow a specific path"]
        T5["🏘️ community_context<br/>Get theme summaries"]
        T6["📊 path_analysis<br/>Analyze path patterns"]
    end

    subgraph Output["Output"]
        O1["📝 Answer"]
        O2["🛤️ Paths Used"]
        O3["📚 Sources"]
        O4["🔍 Reasoning Trace"]
    end

    Q["❓ Complex Question"] --> THINK
    ACT --> Tools
    Tools --> OBSERVE
    ANSWER --> Output

    style THINK fill:#fff2cc
    style ACT fill:#dae8fc
    style OBSERVE fill:#d5e8d4
    style ANSWER fill:#f8cecc
```

### 7.2 PathRAG Agent Tools

```python
PATHRAG_AGENT_TOOLS = [
    {
        "name": "vector_search",
        "description": "Search for document chunks semantically similar to a query",
        "parameters": {
            "query": "Search query text",
            "top_k": "Number of results (default: 5)"
        }
    },
    {
        "name": "entity_lookup",
        "description": "Look up detailed information about a specific entity",
        "parameters": {
            "entity_name": "Name of the entity to look up"
        }
    },
    {
        "name": "find_paths",
        "description": "Find reasoning paths between two entities in the knowledge graph",
        "parameters": {
            "source_entity": "Starting entity",
            "target_entity": "Target entity (optional)",
            "max_length": "Maximum path length (default: 4)"
        }
    },
    {
        "name": "traverse_path",
        "description": "Follow a specific relationship type from an entity",
        "parameters": {
            "entity": "Starting entity",
            "relationship_type": "Type of relationship to follow",
            "direction": "Direction: 'outgoing', 'incoming', or 'both'"
        }
    },
    {
        "name": "community_context",
        "description": "Get summary of a community/theme the entity belongs to",
        "parameters": {
            "entity": "Entity to get community context for"
        }
    },
    {
        "name": "path_analysis",
        "description": "Analyze patterns in paths (common connections, hubs)",
        "parameters": {
            "paths": "List of path IDs to analyze"
        }
    }
]
```

### 7.3 Multi-Hop Reasoning Example

```
QUERY: "What projects did John's collaborators work on, and who approved their budgets?"

┌─────────────────────────────────────────────────────────────────────┐
│ THINK: This is a multi-hop question requiring:                     │
│   1. Find John                                                     │
│   2. Find John's collaborators                                     │
│   3. Find projects of each collaborator                            │
│   4. Find budget approvers for each project                        │
│ Let me start by finding John.                                      │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: entity_lookup("John")                                         │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE:                                                           │
│   Entity: John Smith                                               │
│   Type: PERSON                                                     │
│   Role: Senior Engineer                                            │
│   Organization: Microsoft Azure Team                               │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: Found John Smith. Now I need to find paths to his           │
│        collaborators and their projects.                           │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: find_paths("John Smith", max_length=3)                        │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE: Paths found:                                              │
│   Path 1: John Smith → COLLABORATES → Sarah Johnson                │
│   Path 2: John Smith → COLLABORATES → Mike Lee                     │
│   Path 3: John Smith → COLLABORATES → Sarah → WORKS_ON → Copilot   │
│   Path 4: John Smith → COLLABORATES → Mike → WORKS_ON → AKS        │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: I found collaborators (Sarah, Mike) and their projects      │
│        (Copilot, AKS). Now I need budget approvers.                │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: traverse_path("Azure Copilot", "BUDGET_APPROVED_BY", "incoming")│
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE:                                                           │
│   Azure Copilot ← BUDGET_APPROVED_BY ← Lisa Chen (Finance Dir)     │
├─────────────────────────────────────────────────────────────────────┤
│ ACT: traverse_path("AKS", "BUDGET_APPROVED_BY", "incoming")        │
├─────────────────────────────────────────────────────────────────────┤
│ OBSERVE:                                                           │
│   AKS ← BUDGET_APPROVED_BY ← David Park (VP Engineering)           │
├─────────────────────────────────────────────────────────────────────┤
│ THINK: I now have complete information:                            │
│   - John's collaborators: Sarah Johnson, Mike Lee                  │
│   - Sarah's project: Azure Copilot (approved by Lisa Chen)         │
│   - Mike's project: AKS (approved by David Park)                   │
│   I can now provide the final answer.                              │
├─────────────────────────────────────────────────────────────────────┤
│ ANSWER:                                                            │
│                                                                    │
│ John Smith's collaborators and their projects:                     │
│                                                                    │
│ 1. **Sarah Johnson** works on **Azure Copilot**                    │
│    - Budget approved by: Lisa Chen (Finance Director)              │
│                                                                    │
│ 2. **Mike Lee** works on **Azure Kubernetes Service (AKS)**        │
│    - Budget approved by: David Park (VP Engineering)               │
│                                                                    │
│ REASONING PATHS USED:                                              │
│ - John Smith → COLLABORATES → Sarah Johnson → WORKS_ON → Copilot   │
│ - John Smith → COLLABORATES → Mike Lee → WORKS_ON → AKS            │
│ - Azure Copilot ← BUDGET_APPROVED_BY ← Lisa Chen                   │
│ - AKS ← BUDGET_APPROVED_BY ← David Park                            │
│                                                                    │
│ SOURCES: [email_thread_123, project_doc_456, budget_approval_789]  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Updated Project Timeline

```mermaid
gantt
    title Enhanced Pipeline Timeline (12 Weeks)
    dateFormat  YYYY-MM-DD

    section Phase 1: Ingestion
    Azure Setup & Bronze Layer    :p1a, 2026-01-06, 5d
    Document Extraction           :p1b, after p1a, 5d

    section Phase 2: Silver + ReAct
    Basic NER & Chunking          :p2a, after p1b, 4d
    ReAct Anonymization           :p2b, after p2a, 5d
    ReAct Summarization           :p2c, after p2b, 5d

    section Phase 3: Embeddings
    Embedding Generation          :p3a, after p2c, 3d
    HNSW Index Setup              :p3b, after p3a, 3d
    Baseline RAG                  :p3c, after p3b, 4d

    section Phase 4: Graph + PathRAG
    Entity Extraction             :p4a, after p3c, 5d
    Knowledge Graph Building      :p4b, after p4a, 4d
    Community Detection           :p4c, after p4b, 3d
    PathRAG Implementation        :p4d, after p4c, 5d

    section Phase 5: ReAct Agent
    ReAct Agent + PathRAG Tools   :p5a, after p4d, 5d
    Multi-hop QA Optimization     :p5b, after p5a, 4d

    section Phase 6: Evaluation
    4-System Benchmark            :p6a, after p5b, 4d
    Statistical Analysis          :p6b, after p6a, 3d
    Final Report                  :p6c, after p6b, 3d
```

---

## 9. Cost Estimation

### 9.1 Processing Costs

| Component | Volume | Unit Cost | Total |
|-----------|--------|-----------|-------|
| **Bronze Layer** | | | |
| PST Extraction | 10 GB | - | $0 |
| Document Parsing | 5 GB | - | $0 |
| **Silver Layer** | | | |
| Chunking | 100K chunks | - | $0 |
| Basic NER (spaCy) | 100K chunks | - | $0 |
| Basic PII (Presidio) | 100K chunks | - | $0 |
| ReAct Anonymization | 5K chunks (5%) | $0.05/chunk | $250 |
| ReAct Summarization | 10K chunks (10%) | $0.03/chunk | $300 |
| **Gold Layer** | | | |
| Embeddings | 100K chunks | $0.0001/chunk | $10 |
| Entity Extraction | 100K chunks | $0.004/chunk | $400 |
| PathRAG Index | 10K paths | $0.01/path | $100 |
| **Total Processing** | | | **$1,060** |

### 9.2 Retrieval Costs

| Component | Volume | Unit Cost | Total |
|-----------|--------|-----------|-------|
| Baseline RAG Testing | 500 queries | $0.02/query | $10 |
| GraphRAG Testing | 500 queries | $0.05/query | $25 |
| PathRAG Testing | 500 queries | $0.06/query | $30 |
| PathRAG+ReAct Testing | 500 queries | $0.10/query | $50 |
| **Total Retrieval** | | | **$115** |

### 9.3 Total Budget

| Category | Cost |
|----------|------|
| Processing | $1,060 |
| Retrieval Testing | $115 |
| Buffer (20%) | $235 |
| **Grand Total** | **~$1,400** |

---

## 10. Implementation Checklist

### Phase 1-2: Data Ingestion + ReAct Preprocessing

- [ ] Set up Azure infrastructure (Databricks, ADLS, Key Vault)
- [ ] Implement PST extraction pipeline
- [ ] Implement document parsing (PDF, DOCX, XLSX, PPTX)
- [ ] Create semantic chunking with overlap
- [ ] Implement basic NER with spaCy (en_core_web_trf, nl_core_news_lg)
- [ ] Implement basic PII detection with Presidio
- [ ] Build confidence scoring system
- [ ] Create ReAct agent for anonymization
- [ ] Create ReAct agent for summarization
- [ ] Test on sample dataset (1000 chunks)
- [ ] Measure quality improvement vs baseline

### Phase 3: Vector Index

- [ ] Generate embeddings with text-embedding-3-large
- [ ] Create HNSW index in Azure AI Search
- [ ] Implement baseline RAG retriever
- [ ] Create evaluation dataset
- [ ] Measure baseline metrics

### Phase 4: Knowledge Graph + PathRAG

- [ ] Extract entities with GPT-4o
- [ ] Build knowledge graph in Cosmos DB
- [ ] Implement Leiden community detection
- [ ] Generate community summaries
- [ ] Implement path finding algorithm
- [ ] Implement path scoring function
- [ ] Implement path pruning
- [ ] Create PathRAG retriever
- [ ] Test PathRAG vs GraphRAG

### Phase 5: ReAct Agent

- [ ] Define PathRAG agent tools
- [ ] Build ReAct agent with LangGraph
- [ ] Implement reasoning trace capture
- [ ] Test on multi-hop questions
- [ ] Optimize for latency

### Phase 6: Evaluation

- [ ] Run RAGAS evaluation on all 4 systems
- [ ] Calculate statistical significance
- [ ] Create comparison visualizations
- [ ] Write final report
- [ ] Prepare thesis chapter

---

## 11. Research Contributions

This enhanced architecture provides three potential research contributions:

### Contribution 1: ReAct-Assisted Data Preprocessing

> "We demonstrate that applying agentic reasoning during data preprocessing (specifically for PII anonymization and document summarization) improves downstream retrieval quality while maintaining cost efficiency through selective application to ambiguous cases."

### Contribution 2: PathRAG Benchmark

> "We provide a systematic comparison of retrieval approaches (Baseline RAG, GraphRAG, PathRAG, PathRAG+ReAct) on enterprise email and document datasets, measuring performance on single-hop, multi-hop, and global queries."

### Contribution 3: End-to-End Quality Analysis

> "We analyze how preprocessing quality (NER accuracy, PII handling, summary quality) impacts final retrieval performance, providing guidelines for quality thresholds at each pipeline stage."

---

*Enhanced GraphRAG + PathRAG + ReAct Pipeline*
*KU Leuven Master Thesis - Muhammad Rafiq*
