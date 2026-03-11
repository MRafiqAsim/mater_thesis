"""
Hybrid Retriever Module

Combines multiple retrieval strategies (PathRAG, GraphRAG, Vector, ReAct)
for comprehensive and accurate question answering.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from prompt_loader import get_prompt, get_section, format_prompt

from .retrieval_tools import RetrievalToolkit, ToolResult
from .react_retriever import ReActRetriever, ReActResult

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"
    PATHRAG = "pathrag"
    GRAPHRAG = "graphrag"
    HYBRID = "hybrid"
    REACT = "react"


@dataclass
class RetrievalResult:
    """Result from any retrieval strategy."""
    query: str
    answer: str
    chunks: List[Dict[str, Any]]
    strategy: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    is_grounded: bool = True
    missing_info: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "chunks": self.chunks,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "is_grounded": self.is_grounded,
            "missing_info": self.missing_info,
        }


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # Strategy weights for fusion
    vector_weight: float = 0.3
    pathrag_weight: float = 0.4
    graphrag_weight: float = 0.3

    # Retrieval parameters
    top_k_per_strategy: int = 10
    final_top_k: int = 10
    min_confidence: float = 0.3

    # Answer generation
    use_llm_answer: bool = True
    answer_model: str = "gpt-4o-mini"
    max_context_chunks: int = 5


class HybridRetriever:
    """
    Hybrid retriever that intelligently combines multiple strategies.

    Supports:
    1. Vector Search: Fast semantic similarity
    2. PathRAG: Multi-hop reasoning through entity paths
    3. GraphRAG: Community-based context retrieval
    4. Hybrid: Weighted fusion of all strategies
    5. ReAct: Autonomous agent with tool use
    """

    def __init__(
        self,
        gold_path: str,
        silver_path: Optional[str] = None,
        config: Optional[HybridConfig] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            gold_path: Path to Gold layer
            silver_path: Path to Silver layer
            config: Retrieval configuration
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None
        self.config = config or HybridConfig()

        # Initialize toolkit
        self.toolkit = RetrievalToolkit(str(gold_path), str(silver_path) if silver_path else None)

        # Initialize ReAct retriever
        self.react_retriever = ReActRetriever(str(gold_path), str(silver_path) if silver_path else None)

        # LLM client for answer generation
        self.llm_client = None
        self._initialize_llm()

        logger.info("HybridRetriever initialized")

    def _initialize_llm(self):
        """Initialize LLM for answer generation."""
        try:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")

            if azure_endpoint and azure_key:
                from openai import AzureOpenAI
                self.llm_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                self.config.answer_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
                return

            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=openai_key)
                return

        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> RetrievalResult:
        """
        Retrieve relevant information for a query.

        Args:
            query: The user's question
            strategy: Which retrieval strategy to use

        Returns:
            RetrievalResult with answer and supporting chunks
        """
        start_time = datetime.now()

        if strategy == RetrievalStrategy.VECTOR:
            result = self._vector_retrieve(query)
        elif strategy == RetrievalStrategy.PATHRAG:
            result = self._pathrag_retrieve(query)
        elif strategy == RetrievalStrategy.GRAPHRAG:
            result = self._graphrag_retrieve(query)
        elif strategy == RetrievalStrategy.REACT:
            result = self._react_retrieve(query)
        else:  # HYBRID
            result = self._hybrid_retrieve(query)

        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result

    def _vector_retrieve(self, query: str) -> RetrievalResult:
        """Pure vector similarity retrieval."""
        tool_result = self.toolkit.vector_search(query, top_k=self.config.top_k_per_strategy)

        chunks = tool_result.data if tool_result.success else []

        # Thread expansion: fetch sibling chunks from same threads
        if chunks:
            chunks = self._expand_by_thread(chunks)

        # Generate answer if configured
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer and chunks:
            answer, is_grounded, missing_info = self._generate_answer(query, chunks[:self.config.max_context_chunks])

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy="vector",
            confidence=self._calculate_confidence(chunks),
            sources=[c.get("chunk_id", "") for c in chunks],
            metadata={"tool_message": tool_result.message},
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _pathrag_retrieve(self, query: str) -> RetrievalResult:
        """PathRAG: Entity path-based retrieval."""
        # Extract entities from query
        entities = self._extract_query_entities(query)

        if len(entities) < 2:
            # Fall back to vector search if not enough entities
            return self._vector_retrieve(query)

        # Find paths
        path_result = self.toolkit.pathrag_search(entities, max_paths=5)

        if not path_result.success or not path_result.data:
            return self._vector_retrieve(query)

        # Collect evidence chunks from paths
        chunk_ids = set()
        for path in path_result.data:
            chunk_ids.update(path.get("evidence_chunks", []))

        # Load chunk details
        chunks = []
        for chunk_id in list(chunk_ids)[:self.config.top_k_per_strategy]:
            chunk_result = self.toolkit.get_chunk_context(chunk_id)
            if chunk_result.success and chunk_result.data:
                chunks.append(chunk_result.data)

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer and chunks:
            # Include path information in context
            path_context = "\n".join([
                f"Path: {p.get('description', '')}"
                for p in path_result.data[:3]
            ])
            answer, is_grounded, missing_info = self._generate_answer(
                query,
                chunks[:self.config.max_context_chunks],
                extra_context=f"Reasoning paths found:\n{path_context}"
            )

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy="pathrag",
            confidence=self._calculate_confidence(chunks),
            sources=[c.get("chunk_id", "") for c in chunks],
            metadata={
                "paths_found": len(path_result.data),
                "entities_used": entities
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _detect_aggregate_type(self, query: str) -> Optional[List[str]]:
        """Detect if query asks for a list of entities and return matching entity types.

        Returns a list of entity types to try (primary + related fallbacks).
        For example, 'projects' → ['PRODUCT', 'DOCUMENT', 'CONCEPT'] because
        emails may label work items differently than the user's terminology.
        """
        import re
        query_lower = query.lower()

        # Map user terms to primary + fallback entity types
        type_patterns = {
            r"(project|initiative|task|work item|topic|subject)": ["PRODUCT", "DOCUMENT", "CONCEPT"],
            r"(product|system|tool|software|application|platform)": ["PRODUCT"],
            r"(people|person|employee|team member|participant|who)": ["PERSON"],
            r"(organization|company|department|team|vendor|supplier|client)": ["ORG"],
            r"(event|meeting|milestone|deadline|conference)": ["EVENT"],
            r"(document|report|file|attachment|specification)": ["DOCUMENT"],
            r"(location|city|country|office|place|region)": ["GPE"],
        }

        # Check for listing/aggregate intent
        if not re.search(r"(list|all|every|what are|provide|show|give me|how many|which)", query_lower):
            return None

        for pattern, entity_types in type_patterns.items():
            if re.search(pattern, query_lower):
                return entity_types

        return None

    def _graphrag_retrieve(self, query: str) -> RetrievalResult:
        """GraphRAG: Community-based retrieval with entity listing for aggregate queries."""
        # Check for aggregate/listing queries
        aggregate_type = self._detect_aggregate_type(query)
        extra_context = ""

        if aggregate_type:
            # List entities for each type (primary + fallbacks)
            for etype in aggregate_type:
                entity_list_result = self.toolkit.list_entities(etype)
                if entity_list_result.success and entity_list_result.data:
                    entity_names = [e["name"] for e in entity_list_result.data]
                    extra_context += (
                        f"{etype} entities in the knowledge graph "
                        f"({len(entity_names)} total):\n"
                        + "\n".join(f"- {name}" for name in entity_names)
                        + "\n\n"
                    )

            # Also include thread subjects as work items / projects
            if self.silver_path:
                thread_subjects = self._get_all_thread_subjects()
                if thread_subjects:
                    extra_context += (
                        f"All email thread subjects (work items/topics discussed, "
                        f"{len(thread_subjects)} total):\n"
                        + "\n".join(f"- {s}" for s in sorted(thread_subjects))
                        + "\n\n"
                    )

        # Search communities
        community_result = self.toolkit.graphrag_search(query, level=0, top_k=3)

        if not community_result.success or not community_result.data:
            if not extra_context:
                return self._vector_retrieve(query)

        # Collect chunks from community source_chunk_ids (or fallback to entity lookup)
        chunks = []
        community_summaries = []

        if community_result.success and community_result.data:
            for community in community_result.data:
                community_summaries.append(community.get("summary", ""))

                # Prefer direct source_chunk_ids from community
                source_chunks = community.get("source_chunk_ids", [])
                if source_chunks:
                    for chunk_id in source_chunks[:10]:
                        chunk_result = self.toolkit.get_chunk_context(chunk_id)
                        if chunk_result.success and chunk_result.data:
                            chunks.append(chunk_result.data)
                else:
                    # Fallback: get chunks via entity lookup (for old community files)
                    for entity in community.get("key_entities", [])[:3]:
                        entity_result = self.toolkit.entity_lookup(entity.get("name", ""))
                        if entity_result.success and entity_result.data:
                            for chunk_id in entity_result.data.get("source_chunks", [])[:3]:
                                chunk_result = self.toolkit.get_chunk_context(chunk_id)
                                if chunk_result.success and chunk_result.data:
                                    chunks.append(chunk_result.data)

        # Deduplicate chunks
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        chunks = unique_chunks[:self.config.top_k_per_strategy]

        # Generate answer with community context + entity listing
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer:
            community_context = extra_context + "\n\n".join([
                f"Community Context: {s}" for s in community_summaries
            ])
            answer, is_grounded, missing_info = self._generate_answer(
                query,
                chunks[:self.config.max_context_chunks],
                extra_context=community_context
            )

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy="graphrag",
            confidence=self._calculate_confidence(chunks),
            sources=[c.get("chunk_id", "") for c in chunks],
            metadata={
                "communities_found": len(community_result.data),
                "community_summaries": community_summaries
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _hybrid_retrieve(self, query: str) -> RetrievalResult:
        """Hybrid: Weighted fusion of all strategies."""
        # Run all strategies
        vector_result = self._vector_retrieve(query)
        pathrag_result = self._pathrag_retrieve(query)
        graphrag_result = self._graphrag_retrieve(query)

        # Collect and score chunks
        chunk_scores: Dict[str, Tuple[Dict, float]] = {}

        # Add vector chunks
        for i, chunk in enumerate(vector_result.chunks):
            chunk_id = chunk.get("chunk_id", f"vec_{i}")
            score = self.config.vector_weight * (1.0 - i * 0.1)  # Decay by position
            if chunk.get("similarity_score"):
                score *= chunk["similarity_score"]
            chunk_scores[chunk_id] = (chunk, chunk_scores.get(chunk_id, (chunk, 0))[1] + score)

        # Add PathRAG chunks
        for i, chunk in enumerate(pathrag_result.chunks):
            chunk_id = chunk.get("chunk_id", f"path_{i}")
            score = self.config.pathrag_weight * (1.0 - i * 0.1)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (chunk, chunk_scores[chunk_id][1] + score)
            else:
                chunk_scores[chunk_id] = (chunk, score)

        # Add GraphRAG chunks
        for i, chunk in enumerate(graphrag_result.chunks):
            chunk_id = chunk.get("chunk_id", f"graph_{i}")
            score = self.config.graphrag_weight * (1.0 - i * 0.1)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (chunk, chunk_scores[chunk_id][1] + score)
            else:
                chunk_scores[chunk_id] = (chunk, score)

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top-k
        final_chunks = [chunk for chunk, score in sorted_chunks[:self.config.final_top_k]]

        # Thread expansion: fetch sibling chunks (email body + attachments) from same threads
        final_chunks = self._expand_by_thread(final_chunks)

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer and final_chunks:
            # Include context from all strategies
            extra_context = ""

            # For aggregate queries, include entity listing + thread subjects
            aggregate_type = self._detect_aggregate_type(query)
            if aggregate_type:
                for etype in aggregate_type:
                    entity_list_result = self.toolkit.list_entities(etype)
                    if entity_list_result.success and entity_list_result.data:
                        entity_names = [e["name"] for e in entity_list_result.data]
                        extra_context += (
                            f"{etype} entities ({len(entity_names)} total):\n"
                            + "\n".join(f"- {name}" for name in entity_names)
                            + "\n\n"
                        )
                thread_subjects = self._get_all_thread_subjects()
                if thread_subjects:
                    extra_context += (
                        f"All email thread subjects ({len(thread_subjects)} total):\n"
                        + "\n".join(f"- {s}" for s in sorted(thread_subjects))
                        + "\n\n"
                    )

            if pathrag_result.metadata.get("paths_found", 0) > 0:
                extra_context += f"Found {pathrag_result.metadata['paths_found']} reasoning paths.\n"
            if graphrag_result.metadata.get("community_summaries"):
                extra_context += "Community insights:\n"
                for summary in graphrag_result.metadata["community_summaries"][:2]:
                    extra_context += f"- {summary[:200]}\n"

            answer, is_grounded, missing_info = self._generate_answer(
                query,
                final_chunks[:self.config.max_context_chunks],
                extra_context=extra_context
            )

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=final_chunks,
            strategy="hybrid",
            confidence=self._calculate_confidence(final_chunks),
            sources=[c.get("chunk_id", "") for c in final_chunks],
            metadata={
                "vector_chunks": len(vector_result.chunks),
                "pathrag_chunks": len(pathrag_result.chunks),
                "graphrag_chunks": len(graphrag_result.chunks),
                "fusion_scores": {k: v[1] for k, v in list(chunk_scores.items())[:10]}
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _react_retrieve(self, query: str) -> RetrievalResult:
        """ReAct: Autonomous agent retrieval."""
        react_result = self.react_retriever.query(query)

        # Convert ReAct sources to chunks
        chunks = []
        seen_chunk_ids = set()
        for source in react_result.sources:
            chunk_id = source.get("chunk_id", "")
            if source.get("type") == "chunk" and chunk_id and chunk_id not in seen_chunk_ids:
                chunk_result = self.toolkit.get_chunk_context(chunk_id)
                if chunk_result.success and chunk_result.data:
                    chunks.append(chunk_result.data)
                    seen_chunk_ids.add(chunk_id)

        # Fallback: if the agent produced an answer but no chunk sources
        # (e.g. used only list_entities/graphrag), run a vector search for evidence
        if react_result.answer and not chunks:
            logger.info("ReAct produced answer but no chunk sources — running fallback vector search")
            vector_result = self.toolkit.vector_search(query, top_k=10)
            if vector_result.success and vector_result.data:
                for item in vector_result.data:
                    cid = item.get("chunk_id", "")
                    if cid and cid not in seen_chunk_ids:
                        chunk_result = self.toolkit.get_chunk_context(cid)
                        if chunk_result.success and chunk_result.data:
                            chunks.append(chunk_result.data)
                            seen_chunk_ids.add(cid)

        return RetrievalResult(
            query=query,
            answer=react_result.answer,
            chunks=chunks,
            strategy="react",
            confidence=1.0 if react_result.success else 0.0,
            sources=[s.get("chunk_id", s.get("community_id", s.get("path_id", "")))
                    for s in react_result.sources],
            metadata={
                "steps": len(react_result.steps),
                "total_tokens": react_result.total_tokens,
                "reasoning_trace": [s.to_dict() for s in react_result.steps]
            }
        )

    def _get_all_thread_subjects(self) -> List[str]:
        """Get all unique thread subjects from silver layer."""
        if not self.silver_path:
            return []

        subjects = set()
        silver = Path(self.silver_path)
        for folder in ["thread_chunks", "individual_chunks"]:
            folder_path = silver / folder
            if not folder_path.exists():
                continue
            for f in folder_path.glob("*.json"):
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    subj = data.get("thread_subject", "")
                    if subj:
                        subjects.add(subj)
                except Exception:
                    pass
        return list(subjects)

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract potential entity names from query and match against known graph entities.

        Strategy:
        1. Extract quoted phrases, capitalized words, and acronyms from query
        2. Fuzzy-match all query terms against known graph entity names
        3. Return matched graph entities (enables PathRAG even with lowercase queries)
        """
        import re

        entities = []
        stop_words = {
            # Question words
            'the', 'a', 'an', 'what', 'who', 'where', 'when', 'how', 'why', 'which',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
            # Request verbs
            'about', 'tell', 'me', 'can', 'you', 'find', 'show', 'get', 'give',
            'provide', 'list', 'all', 'every', 'any', 'some',
            # Generic nouns (not entity names)
            'project', 'projects', 'email', 'emails', 'thread', 'threads',
            'information', 'details', 'data', 'discussed', 'mentioned',
            'people', 'person', 'things', 'stuff', 'work', 'used',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by',
        }

        # 1. Quoted phrases (highest priority)
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # 2. Capitalized words and acronyms (any length)
        words = query.split()
        for word in words:
            clean = re.sub(r'[?.!,]$', '', word)
            if clean.lower() in stop_words:
                continue
            if clean.isupper() and len(clean) >= 2:  # Acronyms: GT, SQL, UAT
                entities.append(clean)
            elif clean[0:1].isupper() and len(clean) > 1:
                entities.append(clean)

        # 3. Build multi-word candidates from non-stop words
        content_words = [re.sub(r'[?.!,]$', '', w) for w in words
                        if w.lower() not in stop_words and len(w) > 1]
        if len(content_words) >= 2:
            # Try bigrams and trigrams
            for i in range(len(content_words)):
                for j in range(i + 2, min(i + 4, len(content_words) + 1)):
                    phrase = " ".join(content_words[i:j])
                    entities.append(phrase)
        # Also add individual content words (only if they're meaningful)
        entities.extend(content_words)

        # 4. Match against known graph entities (fuzzy, case-insensitive)
        matched = self._match_graph_entities(list(set(entities)))

        # Return graph-matched entities if found, otherwise raw extractions
        if matched:
            return matched
        return list(set(entities))

    def _match_graph_entities(self, candidates: List[str]) -> List[str]:
        """Match candidate strings against known graph entity names."""
        if not self.gold_path:
            return candidates

        # Load graph entity names (cached)
        if not hasattr(self, '_graph_entity_names'):
            self._graph_entity_names = {}
            graph_file = Path(self.gold_path) / "knowledge_graph" / "nodes.json"
            if graph_file.exists():
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                    for node_id, node in graph_data.items():
                        name = node.get("name", node_id)
                        node_type = node.get("node_type", node.get("type", ""))
                        if name and node_type not in ("CHUNK", "THREAD", ""):
                            self._graph_entity_names[name.lower()] = name
                except Exception:
                    pass

        if not self._graph_entity_names:
            return candidates

        matched = []
        for candidate in candidates:
            candidate_lower = candidate.lower()
            # Exact match
            if candidate_lower in self._graph_entity_names:
                matched.append(self._graph_entity_names[candidate_lower])
                continue
            # Substring match (graph entity contains candidate or vice versa)
            for graph_lower, graph_name in self._graph_entity_names.items():
                if candidate_lower in graph_lower or graph_lower in candidate_lower:
                    matched.append(graph_name)

        return list(set(matched))

    def _expand_by_thread(
        self,
        chunks: List[Dict[str, Any]],
        max_siblings: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval results by fetching sibling chunks and summaries
        from the same thread.

        When a chunk is retrieved, also pull in:
        1. Sibling email/attachment chunks from the same thread
        2. Thread summary (if available)
        3. Attachment summaries (if available)

        This gives the LLM full context regardless of which entry point was hit.
        """
        if not self.silver_path:
            return chunks

        existing_ids = {c.get("chunk_id") for c in chunks}

        # Collect unique thread_ids from results
        thread_ids = set()
        for chunk in chunks:
            tid = chunk.get("thread_id")
            if tid:
                thread_ids.add(tid)

        if not thread_ids:
            return chunks

        # 1. Search Silver layer for sibling chunks
        search_dirs = [
            self.silver_path / "thread_chunks",
            self.silver_path / "individual_chunks",
            self.silver_path / "attachment_chunks" / "knowledge",
        ]

        sibling_chunks = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for chunk_file in search_dir.glob("*.json"):
                if len(sibling_chunks) >= max_siblings:
                    break
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    cid = chunk_data.get("chunk_id")
                    tid = chunk_data.get("thread_id")
                    if tid in thread_ids and cid not in existing_ids:
                        sibling_chunks.append({
                            "chunk_id": cid,
                            "text": chunk_data.get("text_anonymized", ""),
                            "thread_id": tid,
                            "thread_subject": chunk_data.get("thread_subject"),
                            "source_type": chunk_data.get("source_type", "email"),
                            "source_attachment_filename": chunk_data.get("source_attachment_filename", ""),
                            "has_attachments": chunk_data.get("has_attachments", False),
                            "similarity_score": 0.0,
                            "_expanded": True,
                        })
                        existing_ids.add(cid)
                except Exception:
                    continue

        # 2. Load thread summaries for matched threads
        summary_context = []
        thread_summaries_dir = self.silver_path / "thread_summaries"
        if thread_summaries_dir.exists():
            for summary_file in thread_summaries_dir.glob("*.json"):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    if summary_data.get("thread_id") in thread_ids:
                        summary_text = summary_data.get("summary", "")
                        if summary_text:
                            summary_context.append({
                                "chunk_id": f"summary_{summary_data['thread_id']}",
                                "text": f"[Thread Summary] {summary_data.get('subject', '')}: {summary_text}",
                                "thread_id": summary_data["thread_id"],
                                "source_type": "thread_summary",
                                "similarity_score": 0.0,
                                "_expanded": True,
                            })
                            # 3. Follow cross-references to attachment summaries
                            for att_id in summary_data.get("attachment_ids", []):
                                att_summary = self._load_attachment_summary(att_id)
                                if att_summary:
                                    summary_context.append({
                                        "chunk_id": f"att_summary_{att_id}",
                                        "text": f"[Attachment: {att_summary.get('filename', '')}] {att_summary.get('summary', '')}",
                                        "thread_id": summary_data["thread_id"],
                                        "source_type": "attachment_summary",
                                        "similarity_score": 0.0,
                                        "_expanded": True,
                                    })
                except Exception:
                    continue

        expanded_count = len(sibling_chunks) + len(summary_context)
        if expanded_count:
            logger.info(
                f"Thread expansion: added {len(sibling_chunks)} sibling chunks, "
                f"{len(summary_context)} summaries from {len(thread_ids)} threads"
            )

        return list(chunks) + sibling_chunks + summary_context

    def _load_attachment_summary(self, attachment_id: str) -> Optional[Dict[str, Any]]:
        """Load an attachment summary by ID from Silver layer."""
        if not self.silver_path:
            return None
        summary_dir = self.silver_path / "attachment_summaries"
        if not summary_dir.exists():
            return None
        # Try exact match and sanitized match
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in attachment_id)[:100]
        for candidate in (f"{attachment_id}.json", f"{safe_id}.json"):
            path = summary_dir / candidate
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved chunks."""
        if not chunks:
            return 0.0

        # Based on number and quality of chunks
        base_score = min(1.0, len(chunks) / self.config.top_k_per_strategy)

        # Boost if chunks have high similarity scores
        similarity_scores = [c.get("similarity_score", 0.5) for c in chunks]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.5

        return base_score * avg_similarity

    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        extra_context: str = ""
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Generate answer using LLM with grounding check.

        Returns:
            (answer, is_grounded, missing_info)
        """
        if not self.llm_client or not chunks:
            return "", True, None

        # Build context from chunks — label email vs attachment for LLM clarity
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", chunk.get("text_anonymized", ""))[:500]
            chunk_id = chunk.get("chunk_id", "unknown")
            thread = chunk.get("thread_subject", "")
            source_type = chunk.get("source_type", "email")

            if source_type == "attachment":
                filename = chunk.get("source_attachment_filename", "unknown")
                label = f"[{chunk_id}] (Thread: {thread} | Attachment: {filename})"
            else:
                label = f"[{chunk_id}] (Thread: {thread} | Email body)"

            context_parts.append(f"{label}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        if extra_context:
            context = f"{extra_context}\n\n---\n\nEvidence:\n{context}"

        # Format prompts from config/prompts.json
        system_prompt = get_prompt("retrieval", "generation", "system_prompt", "You are a helpful assistant.")
        user_prompt = format_prompt(
            get_prompt("retrieval", "generation", "user_prompt", "Context:\n{context}\n\nQuestion: {question}"),
            context=context,
            question=query,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.answer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=get_prompt("retrieval", "generation", "temperature", 0.3),
                max_tokens=get_prompt("retrieval", "generation", "max_tokens", 1000),
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "", True, None

        # Grounding check
        is_grounded = True
        missing_info: Optional[str] = None

        missing_indicators = get_prompt("retrieval", "generation", "missing_indicators", [
            "don't have enough information",
            "not in the context",
            "cannot find",
            "no information",
            "not mentioned",
        ])
        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                is_grounded = False
                missing_info = "Required information not found in knowledge base"
                break

        return answer, is_grounded, missing_info

    def compare_strategies(self, query: str) -> Dict[str, RetrievalResult]:
        """
        Run all strategies and compare results.

        Useful for evaluation and analysis.
        """
        results = {}

        for strategy in RetrievalStrategy:
            if strategy != RetrievalStrategy.HYBRID:  # Hybrid includes others
                result = self.retrieve(query, strategy)
                results[strategy.value] = result

        return results
