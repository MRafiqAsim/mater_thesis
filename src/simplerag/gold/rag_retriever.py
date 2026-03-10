"""
Gold Layer - RAG Retrieval for User-Facing Queries

Objective: Serve grounded, auditable answers with lineage preservation.

Rules:
- Only serves user-facing queries
- Every output must preserve lineage to Bronze
- Never hallucinate - if information is missing, say so explicitly

Supports: SimpleRAG (vector search), PathRAG (thread context), GraphRAG (relationship context)
"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict

import numpy as np
from openai import AsyncAzureOpenAI

from prompt_loader import get_prompt
from ..silver.processor import SilverRecord
from ..utils.lineage import LineageTracker
from ..utils.config import SimpleRAGConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Gold Layer Data Structures
# =============================================================================

@dataclass
class IndexedChunk:
    """A chunk stored in the Gold index."""
    chunk_id: str
    text: str
    word_count: int

    # Lineage
    email_id: str  # Bronze record ID
    silver_id: str  # Silver record ID
    source_file: str

    # Metadata for enhanced retrieval
    subject: str = ""
    timestamp: str = ""
    sender_email: str = ""
    sender_domain: str = ""
    thread_id: str = ""
    message_id: str = ""
    is_reply: bool = False

    # Summary for embedding-based search (actual text used for LLM context)
    summary: str = ""


@dataclass
class RetrievedChunk:
    """A chunk retrieved from search."""
    chunk_id: str
    text: str
    score: float

    # Lineage
    email_id: str
    silver_id: str
    source_file: str

    # Context
    subject: str = ""
    timestamp: str = ""
    sender: str = ""


@dataclass
class RAGResponse:
    """Response from the RAG system with full lineage."""
    query: str
    answer: str
    sources: List[Dict]
    lineage: List[Dict]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_info: Dict = field(default_factory=dict)

    # Grounding info
    is_grounded: bool = True
    confidence: float = 1.0
    missing_info: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Gold RAG Retriever
# =============================================================================

class GoldRAGRetriever:
    """
    Gold Layer RAG Retriever.

    Provides:
    - Vector-based semantic search (SimpleRAG)
    - Thread-aware retrieval (PathRAG ready)
    - Participant/domain filtering (GraphRAG ready)
    - Grounded answer generation
    - Full lineage tracking
    """

    PROMPT_VERSION = "v1.0"

    def __init__(
        self,
        config: SimpleRAGConfig,
        lineage_tracker: LineageTracker
    ):
        self.config = config
        self.lineage = lineage_tracker
        self.dirs = config.directories

        # Azure OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version
        )

        # In-memory index
        self.chunks: Dict[str, IndexedChunk] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []

        # Secondary indices for PathRAG/GraphRAG
        self.thread_index: Dict[str, List[str]] = {}  # thread_id -> chunk_ids
        self.sender_index: Dict[str, List[str]] = {}  # sender_domain -> chunk_ids

        # Load existing index
        self._load_index()

        logger.info("Gold RAG retriever initialized")

    def _load_index(self):
        """Load existing index from disk."""
        index_path = self.dirs.gold_dir / "index" / "chunks.json"
        embeddings_path = self.dirs.gold_dir / "embeddings" / "embeddings.npz"

        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                for chunk_id, chunk_data in data.items():
                    self.chunks[chunk_id] = IndexedChunk(**chunk_data)
                self.chunk_ids = list(self.chunks.keys())

            # Rebuild secondary indices
            self._rebuild_secondary_indices()

        if embeddings_path.exists():
            data = np.load(embeddings_path, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.chunk_ids = data['ids'].tolist()

        logger.info(f"Loaded index: {len(self.chunks)} chunks")

    def _rebuild_secondary_indices(self):
        """Rebuild thread and sender indices."""
        self.thread_index = {}
        self.sender_index = {}

        for chunk_id, chunk in self.chunks.items():
            # Thread index
            if chunk.thread_id:
                if chunk.thread_id not in self.thread_index:
                    self.thread_index[chunk.thread_id] = []
                self.thread_index[chunk.thread_id].append(chunk_id)

            # Sender domain index
            if chunk.sender_domain:
                if chunk.sender_domain not in self.sender_index:
                    self.sender_index[chunk.sender_domain] = []
                self.sender_index[chunk.sender_domain].append(chunk_id)

    def _save_index(self):
        """Save index to disk."""
        # Save chunks
        index_path = self.dirs.gold_dir / "index" / "chunks.json"
        chunks_data = {
            chunk_id: asdict(chunk)
            for chunk_id, chunk in self.chunks.items()
        }
        with open(index_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings
        if self.embeddings is not None:
            embeddings_path = self.dirs.gold_dir / "embeddings" / "embeddings.npz"
            np.savez(
                embeddings_path,
                embeddings=self.embeddings,
                ids=np.array(self.chunk_ids)
            )

        # Save secondary indices
        indices_path = self.dirs.gold_dir / "index" / "secondary_indices.json"
        with open(indices_path, 'w') as f:
            json.dump({
                "thread_index": self.thread_index,
                "sender_index": self.sender_index
            }, f, indent=2)

        logger.info(f"Index saved: {len(self.chunks)} chunks")

    async def index_silver_records(
        self,
        silver_records: List[SilverRecord],
        skip_existing: bool = True
    ):
        """
        Index Silver records into Gold layer.

        Creates embeddings and updates the search index.
        """
        logger.info(f"Indexing {len(silver_records)} Silver records...")

        # Get existing chunk IDs
        existing_chunks: Set[str] = set(self.chunks.keys()) if skip_existing else set()

        new_chunks = []
        new_texts = []
        skipped = 0

        for record in silver_records:
            summary = record.content.summary or ""

            for chunk in record.chunks:
                chunk_id = chunk.chunk_id

                # Skip if already indexed
                if chunk_id in existing_chunks:
                    skipped += 1
                    continue

                # Create indexed chunk with full metadata
                indexed_chunk = IndexedChunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    word_count=chunk.word_count,
                    email_id=record.email_id,
                    silver_id=record.record_id,
                    source_file=record.lineage.source_file,
                    subject=record.metadata.subject,
                    timestamp=record.metadata.timestamp,
                    sender_email=record.participants.sender.email,
                    sender_domain=record.participants.sender.domain,
                    thread_id=record.thread_id,
                    message_id=record.threading.message_id,
                    is_reply=record.threading.is_reply,
                    summary=summary,
                )

                self.chunks[chunk_id] = indexed_chunk
                new_chunks.append(chunk_id)
                # Embed summary for search; fall back to chunk text if no summary
                new_texts.append(summary if summary else chunk.text)

        if skipped > 0:
            logger.info(f"Skipped {skipped} already indexed chunks")

        if not new_texts:
            logger.info("No new chunks to index")
            return

        # Generate embeddings for new chunks
        logger.info(f"Generating embeddings for {len(new_texts)} new chunks...")
        new_embeddings = await self._generate_embeddings(new_texts)

        # Update embeddings array
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = np.array(new_embeddings)
            self.chunk_ids = new_chunks
        else:
            self.embeddings = np.vstack([self.embeddings, np.array(new_embeddings)])
            self.chunk_ids.extend(new_chunks)

        # Rebuild secondary indices
        self._rebuild_secondary_indices()

        # Save to disk
        self._save_index()

        logger.info(f"Indexed {len(new_chunks)} new chunks (total: {len(self.chunks)})")

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI."""
        embeddings = []
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate long texts
            batch = [t[:8000] for t in batch]

            response = await self.openai_client.embeddings.create(
                model=self.config.azure_openai.embedding_deployment,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if (i + batch_size) % 100 == 0:
                logger.info(f"Generated embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")

        return embeddings

    async def query(
        self,
        question: str,
        top_k: int = None,
        thread_id: Optional[str] = None,
        sender_domain: Optional[str] = None
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            thread_id: Filter to specific thread (PathRAG)
            sender_domain: Filter to specific domain (GraphRAG)

        Returns a grounded answer with full lineage tracking.
        """
        top_k = top_k or self.config.top_k
        logger.info(f"Query: {question}")

        # Step 1: Retrieve relevant chunks
        retrieved = await self._retrieve(
            question,
            top_k,
            thread_id=thread_id,
            sender_domain=sender_domain
        )

        if not retrieved:
            return RAGResponse(
                query=question,
                answer="I could not find any relevant information in the knowledge base to answer this question.",
                sources=[],
                lineage=[],
                is_grounded=False,
                confidence=0.0,
                missing_info="No relevant documents found",
                model_info={
                    "model": self.config.azure_openai.model_name,
                    "version": self.config.azure_openai.model_version
                }
            )

        # Step 2: Build context
        context = self._build_context(retrieved)

        # Step 3: Generate answer
        answer, is_grounded, missing_info = await self._generate_answer(question, context)

        # Step 4: Build response with lineage
        response = RAGResponse(
            query=question,
            answer=answer,
            sources=[
                {
                    "chunk_id": r.chunk_id,
                    "text_preview": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                    "score": round(r.score, 4),
                    "subject": r.subject,
                    "timestamp": r.timestamp,
                    "sender": r.sender,
                    "email_id": r.email_id
                }
                for r in retrieved
            ],
            lineage=[
                {
                    "chunk_id": r.chunk_id,
                    "email_id": r.email_id,
                    "silver_id": r.silver_id,
                    "source_file": r.source_file
                }
                for r in retrieved
            ],
            is_grounded=is_grounded,
            confidence=round(sum(r.score for r in retrieved) / len(retrieved), 4),
            missing_info=missing_info,
            model_info={
                "model": self.config.azure_openai.model_name,
                "version": self.config.azure_openai.model_version,
                "prompt_version": self.PROMPT_VERSION
            }
        )

        # Log query
        self._log_query(response)

        return response

    async def _retrieve(
        self,
        question: str,
        top_k: int,
        thread_id: Optional[str] = None,
        sender_domain: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks via vector similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Generate query embedding
        response = await self.openai_client.embeddings.create(
            model=self.config.azure_openai.embedding_deployment,
            input=[question]
        )
        query_embedding = np.array(response.data[0].embedding)

        # Determine which chunks to search
        if thread_id and thread_id in self.thread_index:
            # PathRAG: Filter to thread
            search_ids = set(self.thread_index[thread_id])
        elif sender_domain and sender_domain in self.sender_index:
            # GraphRAG: Filter to domain
            search_ids = set(self.sender_index[sender_domain])
        else:
            # SimpleRAG: Search all
            search_ids = None

        # Compute similarities
        if search_ids:
            # Filter to specific chunks
            indices = [i for i, cid in enumerate(self.chunk_ids) if cid in search_ids]
            if not indices:
                return []
            filtered_embeddings = self.embeddings[indices]
            filtered_ids = [self.chunk_ids[i] for i in indices]
            similarities = np.dot(filtered_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                chunk_id = filtered_ids[idx]
                score = float(similarities[idx])
                if score >= self.config.similarity_threshold:
                    chunk = self.chunks[chunk_id]
                    results.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        text=chunk.text,
                        score=score,
                        email_id=chunk.email_id,
                        silver_id=chunk.silver_id,
                        source_file=chunk.source_file,
                        subject=chunk.subject,
                        timestamp=chunk.timestamp,
                        sender=chunk.sender_email
                    ))
            return results
        else:
            # Search all embeddings
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                chunk_id = self.chunk_ids[idx]
                score = float(similarities[idx])
                if score >= self.config.similarity_threshold:
                    chunk = self.chunks[chunk_id]
                    results.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        text=chunk.text,
                        score=score,
                        email_id=chunk.email_id,
                        silver_id=chunk.silver_id,
                        source_file=chunk.source_file,
                        subject=chunk.subject,
                        timestamp=chunk.timestamp,
                        sender=chunk.sender_email
                    ))
            return results

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}]"
            if chunk.subject:
                header += f" Subject: {chunk.subject}"
            if chunk.timestamp:
                header += f" | Date: {chunk.timestamp}"
            if chunk.sender:
                header += f" | From: {chunk.sender}"

            parts.append(header)
            parts.append(chunk.text)
            parts.append("")

        return "\n".join(parts)

    async def _generate_answer(
        self,
        question: str,
        context: str
    ) -> tuple[str, bool, Optional[str]]:
        """Generate answer using Azure OpenAI."""
        system_prompt = get_prompt("retrieval", "simplerag_generation", "system_prompt")
        from prompt_loader import format_prompt
        user_prompt = format_prompt(
            get_prompt("retrieval", "simplerag_generation", "user_prompt"),
            context=context,
            question=question,
        )

        response = await self.openai_client.chat.completions.create(
            model=self.config.azure_openai.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=get_prompt("retrieval", "simplerag_generation", "max_tokens", 1000),
            temperature=get_prompt("retrieval", "simplerag_generation", "temperature", 0.3)
        )

        answer = response.choices[0].message.content or ""

        # Check grounding
        missing_indicators = get_prompt("retrieval", "simplerag_generation", "missing_indicators", [
            "don't have enough information",
            "not in the context",
            "cannot find",
            "no information",
            "not mentioned"
        ])

        is_grounded = True
        missing_info = None

        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                is_grounded = False
                missing_info = "Required information not found in knowledge base"
                break

        return answer, is_grounded, missing_info

    def _log_query(self, response: RAGResponse):
        """Log query for audit."""
        log_path = self.dirs.gold_dir / "metadata" / "query_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": response.timestamp,
            "query": response.query,
            "is_grounded": response.is_grounded,
            "confidence": response.confidence,
            "sources_count": len(response.sources),
            "model_info": response.model_info
        }

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def index_all_silver(self, skip_existing: bool = True):
        """Index all Silver records into Gold layer."""
        from ..silver.processor import SilverProcessor

        processor = SilverProcessor(self.config, self.lineage)
        records = processor.list_silver_records()

        if not records:
            logger.warning("No Silver records found to index")
            return

        logger.info(f"Found {len(records)} Silver records to index")
        await self.index_silver_records(records, skip_existing=skip_existing)

    async def rebuild_index(self):
        """Rebuild index from scratch."""
        logger.info("Rebuilding Gold index from scratch...")

        # Clear existing
        self.chunks = {}
        self.embeddings = None
        self.chunk_ids = []
        self.thread_index = {}
        self.sender_index = {}

        # Re-index all
        await self.index_all_silver(skip_existing=False)

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_embeddings": len(self.embeddings) if self.embeddings is not None else 0,
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None and len(self.embeddings) > 0 else 0,
            "unique_threads": len(self.thread_index),
            "unique_domains": len(self.sender_index),
            "unique_emails": len(set(c.email_id for c in self.chunks.values()))
        }
