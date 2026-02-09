# Phase 4: GraphRAG Construction

**Duration:** Weeks 6-8
**Goal:** Build knowledge graph to understand entity relationships

---

## Overview

### What We're Building

In this phase, we transform unstructured text into structured knowledge:
1. Extract entities (people, organizations, projects, etc.)
2. Extract relationships between entities
3. Build a knowledge graph
4. Detect communities (clusters of related entities)
5. Generate summaries for each community

### Why This Matters

**Basic RAG Problem:**
```
Question: "Who worked on Project Alpha with John?"

Basic RAG:
- Finds chunks mentioning "John" OR "Project Alpha"
- No understanding that John WORKS_ON Project Alpha
- No understanding of WHO_ELSE works on Project Alpha
- Answer: Incomplete or wrong

GraphRAG:
- Knows: John --[WORKS_ON]--> Project Alpha
- Knows: Sarah --[WORKS_ON]--> Project Alpha
- Knows: John --[COLLABORATES_WITH]--> Sarah
- Answer: "Sarah worked on Project Alpha with John"
```

### The GraphRAG Advantage

| Capability | Basic RAG | GraphRAG |
|------------|-----------|----------|
| Find facts in text | Yes | Yes |
| Understand relationships | No | Yes |
| Answer "who knows who" | No | Yes |
| Summarize themes | No | Yes |
| Multi-hop reasoning | Limited | Yes |

---

## Prerequisites

### From Phase 3

- [ ] Chunks stored in Silver layer (`/mnt/datalake/silver/chunks/`)
- [ ] Embeddings generated and indexed
- [ ] Basic RAG pipeline working

### Azure Resources

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| Azure OpenAI | Entity extraction | GPT-4o deployment |
| Azure Cosmos DB | Graph storage | Gremlin API |
| Azure Databricks | Processing | Premium tier |

### Python Dependencies

```python
# Add to requirements.txt
networkx>=3.0           # Graph algorithms
igraph>=0.10.0          # Leiden algorithm
leidenalg>=0.10.0       # Community detection
pydantic>=2.0           # Structured outputs
```

---

## Step 1: Entity and Relationship Extraction

### What We're Doing

Using GPT-4o to identify entities and their relationships in each chunk.

### Why

- **Structured Knowledge**: Convert free text to graph nodes and edges
- **Type Safety**: Pydantic ensures consistent entity/relationship types
- **LLM Intelligence**: GPT-4o understands context, not just keywords

### How Entity Extraction Works

```
Input Chunk:
"John Smith joined Microsoft in 2020 and has been leading
the Azure OpenAI project. He frequently collaborates with
Sarah Johnson from the Research team."

Extracted Entities:
┌─────────────────┬────────────┬─────────────────────────┐
│ Name            │ Type       │ Description             │
├─────────────────┼────────────┼─────────────────────────┤
│ John Smith      │ PERSON     │ Azure OpenAI lead       │
│ Microsoft       │ ORG        │ Technology company      │
│ Azure OpenAI    │ PROJECT    │ AI project at Microsoft │
│ Sarah Johnson   │ PERSON     │ Research team member    │
│ Research        │ DEPARTMENT │ Team at Microsoft       │
└─────────────────┴────────────┴─────────────────────────┘

Extracted Relationships:
┌─────────────────┬──────────────────┬─────────────────┐
│ Source          │ Relationship     │ Target          │
├─────────────────┼──────────────────┼─────────────────┤
│ John Smith      │ WORKS_AT         │ Microsoft       │
│ John Smith      │ LEADS            │ Azure OpenAI    │
│ John Smith      │ COLLABORATES     │ Sarah Johnson   │
│ Sarah Johnson   │ MEMBER_OF        │ Research        │
└─────────────────┴──────────────────┴─────────────────┘
```

### Instructions

1. **Run the Entity Extraction Notebook**

   ```
   notebooks/04_graphrag/01_entity_extraction.py
   ```

2. **Understanding the Pydantic Schema**

   ```python
   # This ensures GPT-4o returns structured data

   class Entity(BaseModel):
       name: str              # "John Smith"
       type: EntityType       # PERSON, ORG, PROJECT, etc.
       description: str       # Brief description

   class Relationship(BaseModel):
       source: str           # "John Smith"
       target: str           # "Microsoft"
       type: RelationType    # WORKS_AT, LEADS, etc.
       description: str      # "joined in 2020"

   class ExtractionResult(BaseModel):
       entities: List[Entity]
       relationships: List[Relationship]
   ```

3. **The Extraction Prompt**

   ```python
   EXTRACTION_PROMPT = """
   Extract entities and relationships from this text.

   Entity Types: PERSON, ORGANIZATION, PROJECT, TECHNOLOGY,
                 LOCATION, DATE, EVENT, DOCUMENT

   Relationship Types: WORKS_AT, LEADS, COLLABORATES_WITH,
                       REPORTS_TO, USES, MENTIONS, LOCATED_IN

   Text:
   {text}

   Return structured JSON matching the schema.
   """
   ```

4. **Processing in Batches**

   ```python
   # To control costs and handle rate limits

   from src.graphrag.entity_extractor import EntityExtractor

   extractor = EntityExtractor(
       model="gpt-4o",
       batch_size=50,           # Chunks per batch
       max_concurrent=10,       # Parallel requests
       retry_attempts=3
   )

   # Process all chunks
   results = []
   for batch in chunk_batches:
       batch_results = extractor.extract_batch(batch)
       results.extend(batch_results)
       save_checkpoint(results)  # Save progress
   ```

### Expected Output

```
Silver Layer:
├── /mnt/datalake/silver/entities_raw/
│   └── Extracted entities (may have duplicates)
├── /mnt/datalake/silver/relationships_raw/
│   └── Extracted relationships
└── /mnt/datalake/silver/extraction_metadata/
    └── Processing statistics

Statistics:
- Chunks processed: 100,000
- Entities extracted: ~250,000 (raw)
- Relationships extracted: ~300,000 (raw)
- Processing time: ~4-6 hours
```

### Entity Schema

| Column | Type | Description |
|--------|------|-------------|
| entity_id | string | Unique identifier |
| name | string | Entity name |
| type | string | Entity type (PERSON, ORG, etc.) |
| description | string | Brief description |
| source_chunk_id | string | Chunk where found |
| confidence | float | Extraction confidence |

### Cost Estimation

```
Chunks: 100,000
Avg tokens per chunk: 500
Avg response tokens: 200

Input cost:  100,000 * 500 / 1000 * $0.005 = $250
Output cost: 100,000 * 200 / 1000 * $0.015 = $300
Total: ~$550 for full extraction

Tip: Test with 1,000 chunks first (~$5.50)
```

---

## Step 2: Build Knowledge Graph

### What We're Doing

Converting extracted entities and relationships into a queryable graph structure.

### Why

- **Fast Traversal**: Graph queries are O(1) for neighbors
- **Path Finding**: Find connections between any two entities
- **Persistence**: Cosmos DB stores graph permanently
- **Scalability**: Can handle millions of nodes/edges

### Knowledge Graph Structure

```
                    ┌─────────────────┐
                    │    Microsoft    │
                    │   (ORGANIZATION)│
                    └────────┬────────┘
                             │
                     WORKS_AT│
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌─────────┐           ┌─────────────┐          ┌─────────────┐
│  John   │           │   Sarah     │          │    Bob      │
│ (PERSON)│           │  (PERSON)   │          │  (PERSON)   │
└────┬────┘           └──────┬──────┘          └─────────────┘
     │                       │
     │ LEADS                 │ MEMBER_OF
     │                       │
     ▼                       ▼
┌───────────────┐     ┌────────────┐
│  Azure OpenAI │     │  Research  │
│   (PROJECT)   │     │   (DEPT)   │
└───────────────┘     └────────────┘
```

### Instructions

1. **Run the Knowledge Graph Notebook**

   ```
   notebooks/04_graphrag/02_knowledge_graph.py
   ```

2. **Entity Deduplication**

   Why: Same entity may be extracted with different names:
   - "John Smith", "John", "J. Smith", "Mr. Smith"

   ```python
   from src.graphrag.graph_builder import GraphBuilder

   builder = GraphBuilder()

   # Deduplicate using embeddings + string similarity
   unique_entities = builder.deduplicate_entities(
       entities_raw,
       similarity_threshold=0.85,  # 85% similar = same entity
       use_embeddings=True         # Semantic similarity
   )

   # Result: 250,000 raw → ~50,000 unique entities
   ```

3. **Graph Construction Options**

   **Option A: Cosmos DB (Recommended for Production)**
   ```python
   # Persistent, scalable, supports Gremlin queries

   builder = GraphBuilder(
       backend="cosmosdb",
       endpoint=dbutils.secrets.get("azure-cosmos", "endpoint"),
       key=dbutils.secrets.get("azure-cosmos", "key"),
       database="knowledge_graph",
       container="enterprise_graph"
   )

   builder.build_graph(entities, relationships)
   ```

   **Option B: In-Memory NetworkX (For Testing)**
   ```python
   # Fast, no cloud costs, but not persistent

   builder = GraphBuilder(backend="networkx")
   G = builder.build_graph(entities, relationships)

   # Save to file for later use
   nx.write_gpickle(G, "/dbfs/mnt/datalake/gold/graph.gpickle")
   ```

4. **Adding Entity Embeddings**

   ```python
   # Embed entity descriptions for semantic search

   from src.graphrag.graph_builder import GraphBuilder

   builder.add_entity_embeddings(
       embedding_model="text-embedding-3-large",
       batch_size=100
   )

   # Now entities can be found by semantic similarity
   ```

### Expected Output

```
Gold Layer:
├── /mnt/datalake/gold/entities/
│   └── Deduplicated entities with embeddings
├── /mnt/datalake/gold/relationships/
│   └── Normalized relationships with entity IDs
└── /mnt/datalake/gold/graph_statistics/
    └── Node counts, edge counts, density

Cosmos DB (or NetworkX file):
- Graph with 50,000 nodes
- Graph with 200,000 edges
- Average node degree: 8

Statistics:
- Unique entities: ~50,000
- Unique relationships: ~200,000
- Graph density: ~0.0001
```

### Graph Statistics to Track

| Metric | Description | Expected |
|--------|-------------|----------|
| Nodes | Total unique entities | 50,000 |
| Edges | Total relationships | 200,000 |
| Avg Degree | Avg connections per entity | 8 |
| Density | How connected (0-1) | 0.0001 |
| Components | Disconnected subgraphs | 100-500 |

---

## Step 3: Community Detection

### What We're Doing

Grouping related entities into communities (clusters) using the Leiden algorithm.

### Why

- **Theme Discovery**: Communities represent topics/themes
- **Global Queries**: "What are the main projects?" → Community summaries
- **Hierarchical**: Multiple levels from fine to coarse
- **Fast**: Leiden is faster and better than Louvain

### How Community Detection Works

```
Before Community Detection:
┌─────────────────────────────────────────────────────┐
│  ●───●───●       ●───●                              │
│    \ /           │   │      ●───●───●               │
│     ●            ●───●        \ │ /                 │
│                                 ●                   │
│  (All nodes look the same)                          │
└─────────────────────────────────────────────────────┘

After Community Detection:
┌─────────────────────────────────────────────────────┐
│  ┌─────────┐   ┌─────────┐   ┌─────────────┐       │
│  │ ●───●───● │ │ ●───●   │   │ ●───●───●   │       │
│  │   \ /   │   │ │   │   │   │   \ │ /     │       │
│  │    ●    │   │ ●───●   │   │     ●       │       │
│  │ Comm. 1 │   │ Comm. 2 │   │  Community 3│       │
│  │ (Sales) │   │ (R&D)   │   │  (Finance)  │       │
│  └─────────┘   └─────────┘   └─────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Multi-Resolution Detection

```
Resolution 2.0 (Fine-grained):
├── Community 1: Azure OpenAI Team (5 people)
├── Community 2: Azure Storage Team (8 people)
├── Community 3: Azure Compute Team (6 people)
└── ... (many small communities)

Resolution 1.0 (Medium):
├── Community A: Azure Division (20 people)
├── Community B: Office Division (15 people)
└── ... (medium communities)

Resolution 0.5 (Coarse):
├── Community X: All Engineering (50 people)
└── Community Y: All Business (30 people)
```

### Instructions

1. **Run the Community Detection Notebook**

   ```
   notebooks/04_graphrag/03_community_detection.py
   ```

2. **Understanding Leiden Algorithm**

   ```python
   from src.graphrag.community_detector import CommunityDetector

   detector = CommunityDetector()

   # Detect at multiple resolutions
   communities = detector.detect_communities(
       graph=G,
       resolutions=[0.5, 1.0, 2.0],  # Coarse → Fine
       algorithm="leiden"
   )

   # Returns hierarchical communities:
   # Level 0 (resolution 0.5): ~50 communities
   # Level 1 (resolution 1.0): ~200 communities
   # Level 2 (resolution 2.0): ~800 communities
   ```

3. **Resolution Parameter Explained**

   | Resolution | Result | Use Case |
   |------------|--------|----------|
   | 0.5 | Large, few communities | "What are the main themes?" |
   | 1.0 | Balanced communities | Default, general use |
   | 2.0 | Small, many communities | "What are the specific topics?" |

4. **Community Assignment**

   ```python
   # Each entity gets assigned to communities at each level

   # Example entity: "John Smith"
   entity_communities = {
       "level_0": "Engineering",           # Coarse
       "level_1": "Azure Division",        # Medium
       "level_2": "Azure OpenAI Team"      # Fine
   }
   ```

### Expected Output

```
Gold Layer:
├── /mnt/datalake/gold/communities/
│   ├── level_0/  (resolution 0.5)
│   ├── level_1/  (resolution 1.0)
│   └── level_2/  (resolution 2.0)
└── /mnt/datalake/gold/community_members/
    └── Entity-to-community mappings

Statistics by Level:
- Level 0: 50 communities, avg 1000 members
- Level 1: 200 communities, avg 250 members
- Level 2: 800 communities, avg 60 members
```

### Community Schema

| Column | Type | Description |
|--------|------|-------------|
| community_id | string | Unique community identifier |
| level | int | Hierarchy level (0, 1, 2) |
| resolution | float | Leiden resolution parameter |
| member_count | int | Number of entities |
| entity_ids | array | Member entity IDs |
| top_entities | array | Most connected members |

---

## Step 4: Community Summarization

### What We're Doing

Generating natural language summaries for each community using GPT-4o.

### Why

- **Global Queries**: Answer "What are the main themes?" without reading all documents
- **Context Compression**: 1000 entities → 1 summary paragraph
- **Theme Identification**: Discover what each community is about
- **Hierarchical Understanding**: Summaries at different granularities

### How Community Summarization Works

```
Community: "Azure OpenAI Team"
Members: John Smith (Lead), Sarah Johnson, Mike Lee, ...
Relationships: WORKS_ON, COLLABORATES_WITH, REPORTS_TO

GPT-4o Generates:
┌─────────────────────────────────────────────────────────────┐
│ COMMUNITY SUMMARY                                           │
├─────────────────────────────────────────────────────────────┤
│ The Azure OpenAI Team is a core engineering group at        │
│ Microsoft focused on developing AI services. Led by John    │
│ Smith, the team collaborates closely with the Research      │
│ division. Key projects include GPT integration and the      │
│ Copilot initiative. The team works primarily with Python    │
│ and Azure infrastructure.                                   │
│                                                             │
│ Key Entities: John Smith, Sarah Johnson, Azure Copilot      │
│ Key Themes: AI development, Cloud services, LLM integration │
└─────────────────────────────────────────────────────────────┘
```

### Instructions

1. **Run the Community Summarization Notebook**

   ```
   notebooks/04_graphrag/04_community_summarization.py
   ```

2. **The Summarization Process**

   ```python
   from src.graphrag.community_summarizer import CommunitySummarizer

   summarizer = CommunitySummarizer(
       model="gpt-4o",
       max_entities_per_summary=50,  # Limit context size
       include_relationships=True
   )

   # For each community:
   for community in communities:
       # 1. Get community members
       members = get_community_members(community.id)

       # 2. Get internal relationships
       relationships = get_internal_relationships(community.id)

       # 3. Generate summary
       summary = summarizer.summarize(
           entities=members,
           relationships=relationships,
           level=community.level
       )

       # 4. Save summary
       save_summary(community.id, summary)
   ```

3. **The Summarization Prompt**

   ```python
   COMMUNITY_SUMMARY_PROMPT = """
   Summarize this community of related entities.

   Community Members:
   {entity_list}

   Relationships:
   {relationship_list}

   Write a 2-3 paragraph summary that:
   1. Describes what this community is about
   2. Identifies the key entities and their roles
   3. Explains the main themes or topics
   4. Notes important relationships

   Also provide:
   - Top 5 key entities
   - Top 3 themes
   """
   ```

4. **Indexing Summaries for Search**

   ```python
   # Embed summaries for semantic search

   from src.indexing.embedding_generator import EmbeddingGenerator

   embedder = EmbeddingGenerator()

   for summary in community_summaries:
       # Generate embedding
       embedding = embedder.generate(summary.text)

       # Add to search index
       add_to_index(
           id=summary.community_id,
           text=summary.text,
           embedding=embedding,
           metadata={
               "type": "community_summary",
               "level": summary.level,
               "member_count": summary.member_count
           }
       )
   ```

### Expected Output

```
Gold Layer:
├── /mnt/datalake/gold/community_summaries/
│   ├── level_0_summaries/  (50 summaries)
│   ├── level_1_summaries/  (200 summaries)
│   └── level_2_summaries/  (800 summaries)
└── /mnt/datalake/gold/community_summaries_indexed/
    └── Summaries with embeddings

Azure AI Search:
- New index: "community-summaries"
- Documents: 1,050 summaries (all levels)
```

### Community Summary Schema

| Column | Type | Description |
|--------|------|-------------|
| community_id | string | Community identifier |
| level | int | Hierarchy level |
| summary | string | Generated summary text |
| key_entities | array | Top 5 important entities |
| key_themes | array | Top 3 themes |
| member_count | int | Number of entities |
| embedding | array | Summary embedding vector |

### Cost Estimation

```
Communities: 1,050 (all levels)
Avg input tokens: 2,000 (entities + relationships)
Avg output tokens: 500 (summary)

Input cost:  1,050 * 2,000 / 1000 * $0.005 = $10.50
Output cost: 1,050 * 500 / 1000 * $0.015  = $7.88
Total: ~$20 for all summaries
```

---

## Phase 4 Checklist

Before moving to Phase 5, verify:

- [ ] Entities extracted
  - [ ] All chunks processed
  - [ ] Entity types are valid
  - [ ] Descriptions are meaningful

- [ ] Relationships extracted
  - [ ] Source/target are valid entities
  - [ ] Relationship types make sense
  - [ ] Descriptions add context

- [ ] Knowledge graph built
  - [ ] Entities deduplicated
  - [ ] Graph is queryable
  - [ ] Statistics look reasonable

- [ ] Communities detected
  - [ ] Multiple resolution levels
  - [ ] Community sizes are reasonable
  - [ ] Hierarchical structure makes sense

- [ ] Community summaries generated
  - [ ] All communities have summaries
  - [ ] Summaries are indexed
  - [ ] Can search summaries semantically

---

## Verification Queries

```python
# Check entity extraction
entities_df = spark.read.format("delta").load("/mnt/datalake/gold/entities")
print(f"Unique entities: {entities_df.count()}")
entities_df.groupBy("type").count().show()

# Check relationships
rels_df = spark.read.format("delta").load("/mnt/datalake/gold/relationships")
print(f"Relationships: {rels_df.count()}")
rels_df.groupBy("type").count().show()

# Check communities
communities_df = spark.read.format("delta").load("/mnt/datalake/gold/communities")
communities_df.groupBy("level").agg(
    count("*").alias("num_communities"),
    avg("member_count").alias("avg_size")
).show()

# Check summaries
summaries_df = spark.read.format("delta").load("/mnt/datalake/gold/community_summaries")
print(f"Total summaries: {summaries_df.count()}")
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Too many duplicate entities | Low similarity threshold | Increase threshold to 0.9 |
| Communities too large | Low resolution | Increase resolution to 2.0 |
| Communities too small | High resolution | Decrease resolution to 0.5 |
| Extraction timeouts | Large chunks | Reduce chunk size or increase timeout |
| Empty relationships | Chunks too short | Require minimum chunk length |
| Cosmos DB throttling | Too many requests | Reduce batch size, add delays |

---

## Cost Summary for Phase 4

| Step | Estimated Cost |
|------|---------------|
| Entity extraction (100K chunks) | $550 |
| Entity embeddings (50K entities) | $5 |
| Community summarization | $20 |
| Cosmos DB (optional) | $50/month |
| **Total** | **~$625** |

---

## What's Next

In **Phase 5: ReAct Agent**, we will:
1. Build a combined retriever (vector + graph + community)
2. Create an intelligent agent with reasoning capabilities
3. Test multi-hop question answering
4. Compare all retrieval approaches

---

*Phase 4 Complete! Proceed to [Phase 5: ReAct Agent](./PHASE_5_REACT_AGENT.md)*
