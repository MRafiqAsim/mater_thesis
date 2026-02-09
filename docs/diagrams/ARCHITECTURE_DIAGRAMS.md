# Architecture Diagrams

This document contains architecture diagrams for the GraphRAG + ReAct Knowledge Retrieval System.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Sources["📁 DATA SOURCES"]
        PST["📧 PST Files<br/>35 years emails"]
        PDF["📄 PDF Documents"]
        DOCX["📝 DOCX Files"]
        XLSX["📊 XLSX Spreadsheets"]
        PPTX["📽️ PPTX Presentations"]
    end

    subgraph Bronze["🥉 BRONZE LAYER<br/>Raw Data"]
        B1["bronze/emails/"]
        B2["bronze/documents/"]
        B3["bronze/attachments/"]
        B4["bronze/metadata/"]
    end

    subgraph Silver["🥈 SILVER LAYER<br/>Cleaned & Enriched"]
        S1["silver/chunks/"]
        S2["silver/chunks_anonymized/"]
        S3["silver/ner_results/"]
        S4["silver/summaries/"]
        S5["silver/entities_raw/"]
    end

    subgraph Gold["🥇 GOLD LAYER<br/>Business-Ready"]
        G1["gold/entities/"]
        G2["gold/relationships/"]
        G3["gold/communities/"]
        G4["gold/community_summaries/"]
        G5["gold/chunks_embedded/"]
        G6["gold/evaluation_results/"]
    end

    subgraph Azure["☁️ AZURE SERVICES"]
        AOA["🤖 Azure OpenAI<br/>GPT-4o + Embeddings"]
        AIS["🔍 Azure AI Search<br/>HNSW Vector Index"]
        CDB["🌐 Cosmos DB<br/>Gremlin Graph"]
        ADB["⚡ Databricks<br/>Spark Processing"]
    end

    Sources --> Bronze
    Bronze --> Silver
    Silver --> Gold
    Gold <--> Azure

    style Bronze fill:#CD7F32,stroke:#8B4513,color:#fff
    style Silver fill:#C0C0C0,stroke:#808080,color:#000
    style Gold fill:#FFD700,stroke:#B8860B,color:#000
    style Azure fill:#dae8fc,stroke:#6c8ebf
```

---

## 2. Medallion Architecture Detail

```mermaid
flowchart LR
    subgraph Phase1["PHASE 1-2"]
        direction TB
        I1["PST Extraction"]
        I2["Document Parsing"]
        I3["Language Detection"]
        I4["Semantic Chunking"]
        I5["NER + PII"]
        I6["Summarization"]
    end

    subgraph Phase3["PHASE 3-4"]
        direction TB
        P1["Embeddings<br/>text-embedding-3-large"]
        P2["HNSW Indexing"]
        P3["Entity Extraction<br/>GPT-4o"]
        P4["Graph Building<br/>Cosmos DB"]
        P5["Community Detection<br/>Leiden"]
        P6["Community Summaries"]
    end

    subgraph Phase5["PHASE 5-6"]
        direction TB
        A1["Query Classification"]
        A2["GraphRAG Retriever"]
        A3["ReAct Agent"]
        A4["RAGAS Evaluation"]
        A5["Statistical Analysis"]
        A6["Final Report"]
    end

    B[("🥉 BRONZE<br/>Raw Data")]
    S[("🥈 SILVER<br/>Processed")]
    G[("🥇 GOLD<br/>Ready")]

    Phase1 --> B
    B --> S
    Phase3 --> S
    S --> G
    Phase5 --> G
```

---

## 3. GraphRAG Retrieval Pipeline

```mermaid
flowchart TB
    Q["❓ User Question"]

    subgraph Classifier["Query Classifier"]
        C1{"Query Type?"}
        LOCAL["LOCAL<br/>Specific facts"]
        GLOBAL["GLOBAL<br/>Themes/summaries"]
        HYBRID["HYBRID<br/>Both"]
    end

    subgraph Retrieval["Retrieval Methods"]
        VS["🔢 Vector Search<br/>Azure AI Search<br/>Semantic similarity"]
        GS["🔗 Graph Search<br/>Entity lookup<br/>Relationships"]
        CS["🏘️ Community Search<br/>Theme summaries<br/>Global context"]
    end

    subgraph Context["Context Builder"]
        CB["Combine + Rank + Deduplicate"]
        CTX["📋 GraphRAGContext<br/>chunks + entities +<br/>relationships + summaries"]
    end

    Q --> C1
    C1 --> LOCAL
    C1 --> GLOBAL
    C1 --> HYBRID

    LOCAL --> VS
    LOCAL --> GS
    GLOBAL --> CS
    HYBRID --> VS
    HYBRID --> GS
    HYBRID --> CS

    VS --> CB
    GS --> CB
    CS --> CB
    CB --> CTX

    style LOCAL fill:#dae8fc
    style GLOBAL fill:#e1d5e7
    style HYBRID fill:#fff2cc
```

---

## 4. ReAct Agent Architecture

```mermaid
flowchart TB
    subgraph Agent["🤖 ReAct Agent"]
        direction TB

        THINK["💭 THINK<br/>Analyze question<br/>Plan next step"]
        ACT["⚡ ACT<br/>Use a tool"]
        OBSERVE["👁️ OBSERVE<br/>Process result"]
        ANSWER["✅ ANSWER<br/>Final response"]

        THINK --> ACT
        ACT --> OBSERVE
        OBSERVE --> THINK
        THINK --> ANSWER
    end

    subgraph Tools["Agent Tools"]
        T1["🔢 vector_search<br/>Find document chunks"]
        T2["👤 entity_lookup<br/>Get entity details"]
        T3["🔗 relationship_search<br/>Find connections"]
        T4["🏘️ community_search<br/>Get theme summaries"]
        T5["🌐 graph_traversal<br/>Explore neighborhood"]
    end

    subgraph Output["Output"]
        ANS["📝 Answer"]
        SRC["📚 Sources"]
        TRC["🔍 Reasoning Trace"]
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

---

## 5. Multi-Hop Question Example

```mermaid
sequenceDiagram
    participant U as User
    participant A as ReAct Agent
    participant T as Tools
    participant L as GPT-4o

    U->>A: "What projects did John's collaborators work on?"

    A->>L: THINK: I need to find John's collaborators first
    A->>T: entity_lookup("John")
    T-->>A: John Smith, Senior Engineer, Azure OpenAI

    A->>L: THINK: Now find John's collaborators
    A->>T: relationship_search("John Smith", "COLLABORATES_WITH")
    T-->>A: Sarah Johnson, Mike Lee

    A->>L: THINK: Now find Sarah's projects
    A->>T: entity_lookup("Sarah Johnson")
    T-->>A: Works on: Azure Copilot, Azure OpenAI

    A->>L: THINK: Now find Mike's projects
    A->>T: entity_lookup("Mike Lee")
    T-->>A: Works on: Azure Kubernetes Service

    A->>L: THINK: I have enough info to answer
    A->>U: "John's collaborators (Sarah, Mike) work on Azure Copilot, Azure OpenAI, and AKS"
```

---

## 6. User Interface

```mermaid
flowchart LR
    subgraph Users["👤 Users"]
        U1["Knowledge Worker"]
        U2["Analyst"]
        U3["Manager"]
        U4["Developer"]
    end

    subgraph Interfaces["🖥️ Interface Options"]
        I1["💬 Chat Interface<br/>Natural Language<br/>Conversational Q&A"]
        I2["🔌 REST API<br/>Programmatic Access<br/>App Integration"]
        I3["📓 Databricks Notebook<br/>Interactive Exploration<br/>Ad-hoc Analysis"]
    end

    subgraph System["⚙️ GraphRAG + ReAct System"]
        S1["Query Classification"]
        S2["Retrieval Pipeline"]
        S3["ReAct Agent"]
    end

    subgraph Response["✅ Response"]
        R1["📝 Answer"]
        R2["📚 Sources"]
        R3["🔍 Reasoning Trace"]
        R4["🔗 Related Entities"]
    end

    Users --> Interfaces
    Interfaces --> System
    System --> Response
    Response --> Users

    style I1 fill:#dae8fc
    style I2 fill:#e1d5e7
    style I3 fill:#fff2cc
```

---

## 7. End-to-End User Interaction

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant UI as 💬 Chat Interface
    participant QC as Query Classifier
    participant GR as GraphRAG Retriever
    participant RA as ReAct Agent
    participant LLM as 🤖 GPT-4o

    U->>UI: "Who worked with John on Azure projects?"
    UI->>QC: Classify query type
    QC-->>UI: HYBRID (specific + relationships)

    UI->>GR: Retrieve context
    Note over GR: Vector Search + Graph Search + Community Search
    GR-->>UI: GraphRAGContext (chunks, entities, relationships)

    UI->>RA: Process with ReAct Agent

    loop ReAct Loop
        RA->>LLM: THINK: What do I need?
        LLM-->>RA: Plan next action
        RA->>GR: ACT: Use tool (entity_lookup, etc.)
        GR-->>RA: OBSERVE: Tool result
    end

    RA->>LLM: Synthesize final answer
    LLM-->>RA: Complete answer with sources

    RA-->>UI: Answer + Sources + Reasoning Trace
    UI-->>U: "Sarah and Mike worked with John on Azure OpenAI..."

    Note over U,LLM: Complete interaction in 2-5 seconds
```

---

## 8. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        D1["10-15 GB<br/>PST + Documents"]
    end

    subgraph P1["Phase 1"]
        A1["Extract Text"]
        A2["Detect Language"]
    end

    subgraph P2["Phase 2"]
        B1["Chunk (512-1024 tokens)"]
        B2["NER (EN/NL)"]
        B3["Anonymize PII"]
        B4["Summarize"]
    end

    subgraph P3["Phase 3"]
        C1["Generate Embeddings"]
        C2["Create HNSW Index"]
        C3["Basic RAG"]
    end

    subgraph P4["Phase 4"]
        D1a["Extract Entities"]
        D2["Build Graph"]
        D3["Detect Communities"]
        D4["Summarize Communities"]
    end

    subgraph P5["Phase 5"]
        E1["Combined Retriever"]
        E2["ReAct Agent"]
        E3["User Interface"]
    end

    Input --> P1 --> P2 --> P3 --> P4 --> P5
```

---

## 9. Knowledge Graph Structure

```mermaid
graph TB
    subgraph Entities["Entity Types"]
        P["👤 PERSON"]
        O["🏢 ORGANIZATION"]
        PR["📁 PROJECT"]
        T["💻 TECHNOLOGY"]
        L["📍 LOCATION"]
        E["📅 EVENT"]
    end

    subgraph Example["Example Graph"]
        John["John Smith<br/>PERSON"]
        MS["Microsoft<br/>ORG"]
        AO["Azure OpenAI<br/>PROJECT"]
        Sarah["Sarah Johnson<br/>PERSON"]
        Copilot["Azure Copilot<br/>PROJECT"]

        John -->|WORKS_AT| MS
        John -->|LEADS| AO
        John -->|COLLABORATES| Sarah
        Sarah -->|WORKS_ON| Copilot
        Sarah -->|WORKS_ON| AO
    end

    subgraph Communities["Community Detection"]
        C1["Community 1<br/>Azure AI Team"]
        C2["Community 2<br/>Research Division"]

        John -.-> C1
        Sarah -.-> C1
        AO -.-> C1
    end
```

---

## 10. Azure Architecture

```mermaid
flowchart TB
    subgraph VNet["Azure Virtual Network"]
        subgraph Compute["Compute"]
            ADB["⚡ Azure Databricks<br/>Premium Tier<br/>Spark Clusters"]
        end

        subgraph Storage["Storage"]
            ADLS["💾 ADLS Gen2<br/>Delta Lake<br/>Bronze/Silver/Gold"]
        end

        subgraph AI["AI Services"]
            AOA["🤖 Azure OpenAI<br/>GPT-4o<br/>text-embedding-3-large"]
            AIS["🔍 Azure AI Search<br/>Standard S1<br/>HNSW Index"]
        end

        subgraph DB["Database"]
            CDB["🌐 Cosmos DB<br/>Gremlin API<br/>Knowledge Graph"]
        end

        subgraph Security["Security"]
            KV["🔐 Key Vault<br/>Secrets Management"]
        end
    end

    ADB <--> ADLS
    ADB <--> AOA
    ADB <--> AIS
    ADB <--> CDB
    ADB <--> KV

    AOA <--> AIS
```

---

## 11. Execution Timeline

```mermaid
gantt
    title Project Execution Timeline (11 Weeks)
    dateFormat  YYYY-MM-DD

    section Phase 1
    Azure Setup           :p1a, 2026-01-06, 3d
    PST Ingestion         :p1b, after p1a, 5d
    Document Ingestion    :p1c, after p1b, 4d
    Language Detection    :p1d, after p1c, 2d

    section Phase 2
    Semantic Chunking     :p2a, after p1d, 3d
    NER Extraction        :p2b, after p2a, 4d
    PII Anonymization     :p2c, after p2b, 3d
    Summarization         :p2d, after p2c, 4d

    section Phase 3
    Embedding Generation  :p3a, after p2d, 2d
    HNSW Indexing         :p3b, after p3a, 2d
    Basic RAG Pipeline    :p3c, after p3b, 3d

    section Phase 4
    Entity Extraction     :p4a, after p3c, 5d
    Knowledge Graph       :p4b, after p4a, 5d
    Community Detection   :p4c, after p4b, 4d
    Community Summaries   :p4d, after p4c, 3d

    section Phase 5
    GraphRAG Retriever    :p5a, after p4d, 4d
    ReAct Agent           :p5b, after p5a, 5d
    Multi-Hop QA          :p5c, after p5b, 3d

    section Phase 6
    RAGAS Evaluation      :p6a, after p5c, 3d
    Statistical Analysis  :p6b, after p6a, 2d
    Final Report          :p6c, after p6b, 2d
```

---

## How to Use These Diagrams

### Draw.io

1. Open [draw.io](https://app.diagrams.net/)
2. File → Open → Select `architecture_diagram.drawio`
3. Edit as needed
4. Export as PNG/SVG for thesis

### Mermaid (This File)

1. These diagrams render automatically in:
   - GitHub markdown preview
   - VS Code with Mermaid extension
   - Obsidian
   - Many documentation tools

2. To export as images:
   - Use [Mermaid Live Editor](https://mermaid.live/)
   - Paste the code
   - Download as PNG/SVG

### For Thesis (LaTeX)

1. Export diagrams as PNG or SVG
2. Include in LaTeX:
   ```latex
   \begin{figure}[h]
       \centering
       \includegraphics[width=0.9\textwidth]{figures/architecture.png}
       \caption{GraphRAG + ReAct System Architecture}
       \label{fig:architecture}
   \end{figure}
   ```

---

*GraphRAG + ReAct Knowledge Retrieval System*
*KU Leuven Master Thesis - Muhammad Rafiq*
