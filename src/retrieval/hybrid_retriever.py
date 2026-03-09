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

import yaml

from .retrieval_tools import RetrievalToolkit, ToolResult
from .react_retriever import ReActRetriever, ReActResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt configuration (loaded from config/prompts.yaml)
# ---------------------------------------------------------------------------
_PROMPTS_CACHE: Optional[Dict[str, Any]] = None


def _load_prompts() -> Dict[str, Any]:
    """Load prompt configuration from config/prompts.yaml (cached)."""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    # Walk up from this file to find the project root (contains config/)
    prompts_path = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            _PROMPTS_CACHE = yaml.safe_load(f)
            logger.info(f"Loaded prompts from {prompts_path}")
            return _PROMPTS_CACHE

    logger.warning(f"Prompts config not found at {prompts_path}, using built-in defaults")
    _PROMPTS_CACHE = {}
    return _PROMPTS_CACHE


def _get_generation_config() -> Dict[str, Any]:
    """Get the 'generation' section from prompts config with defaults."""
    prompts = _load_prompts()
    gen = prompts.get("generation", {})
    defaults = {
        "system_prompt": (
            "You are a helpful assistant that answers questions based ONLY on the provided context.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY use information from the provided context\n"
            "2. If the context doesn't contain the answer, say \"I don't have enough information to answer this question\"\n"
            "3. Always cite which source(s) you used by mentioning [chunk_id] format\n"
            "4. Never make up information or use knowledge from outside the context\n"
            "5. If you can partially answer, do so and explain what information is missing\n\n"
            "Be concise and factual."
        ),
        "user_prompt": (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer based ONLY on the context above. If the information is not in the context, say so explicitly."
        ),
        "temperature": 0.3,
        "max_tokens": 1000,
        "missing_indicators": [
            "don't have enough information",
            "not in the context",
            "cannot find",
            "no information",
            "not mentioned",
            "insufficient context",
            "does not contain",
        ],
    }
    for key, default in defaults.items():
        gen.setdefault(key, default)
    return gen


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

    def _graphrag_retrieve(self, query: str) -> RetrievalResult:
        """GraphRAG: Community-based retrieval."""
        # Search communities
        community_result = self.toolkit.graphrag_search(query, level=0, top_k=3)

        if not community_result.success or not community_result.data:
            return self._vector_retrieve(query)

        # Collect chunks from community entities
        chunks = []
        community_summaries = []

        for community in community_result.data:
            community_summaries.append(community.get("summary", ""))

            # Get chunks for key entities
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

        # Generate answer with community context
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer:
            community_context = "\n\n".join([
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

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        if self.config.use_llm_answer and final_chunks:
            # Include context from all strategies
            extra_context = ""
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

        # Convert ReAct result to standard result
        chunks = []
        for source in react_result.sources:
            if source.get("type") == "chunk":
                chunk_result = self.toolkit.get_chunk_context(source.get("chunk_id", ""))
                if chunk_result.success and chunk_result.data:
                    chunks.append(chunk_result.data)

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

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        # Simple extraction: capitalized words and quoted phrases
        import re

        entities = []

        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            # Skip common words
            if word.lower() in {'the', 'a', 'an', 'what', 'who', 'where', 'when', 'how', 'why',
                               'did', 'does', 'do', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}:
                continue
            if word[0].isupper() and len(word) > 2:
                entities.append(word)

        return list(set(entities))

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

        gen_cfg = _get_generation_config()

        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", chunk.get("text_anonymized", ""))[:500]
            chunk_id = chunk.get("chunk_id", "unknown")
            thread = chunk.get("thread_subject", "")
            context_parts.append(f"[{chunk_id}] (Thread: {thread})\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        if extra_context:
            context = f"{extra_context}\n\n---\n\nEvidence:\n{context}"

        # Format prompts from config
        system_prompt = gen_cfg["system_prompt"]
        user_prompt = gen_cfg["user_prompt"].format(
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
                temperature=gen_cfg["temperature"],
                max_tokens=gen_cfg["max_tokens"],
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "", True, None

        # Grounding check
        is_grounded = True
        missing_info: Optional[str] = None

        for indicator in gen_cfg["missing_indicators"]:
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
