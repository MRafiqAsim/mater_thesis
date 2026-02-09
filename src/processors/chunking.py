"""
Semantic Chunking Module
========================
Intelligent document chunking using LangChain with Azure OpenAI embeddings.

Strategies:
1. Semantic Chunking - Split based on embedding similarity
2. Recursive Character - Fallback with overlap
3. Document-aware - Respect document structure (headers, paragraphs)

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import hashlib

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    token_count: int
    parent_document_id: str
    chunking_strategy: str


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    # Semantic chunking
    use_semantic: bool = True
    breakpoint_threshold_type: str = "percentile"  # percentile, standard_deviation, interquartile
    breakpoint_threshold_amount: float = 95.0

    # Fallback recursive chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Token limits
    max_tokens_per_chunk: int = 512
    min_tokens_per_chunk: int = 50

    # Model settings
    embedding_model: str = "text-embedding-3-large"


class SemanticChunker:
    """
    Semantic chunking using Azure OpenAI embeddings.

    Splits text at natural semantic boundaries by detecting
    significant changes in embedding similarity.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[ChunkingConfig] = None
    ):
        """
        Initialize semantic chunker.

        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()

        # Initialize Azure OpenAI embeddings
        from langchain_openai import AzureOpenAIEmbeddings

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.embedding_model,
        )

        # Initialize semantic splitter
        if self.config.use_semantic:
            from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

            self.semantic_splitter = LCSemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            )

        # Fallback splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Token counter
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def chunk_document(
        self,
        document: Document,
        use_semantic: bool = True
    ) -> List[Document]:
        """
        Chunk a single document.

        Args:
            document: LangChain Document to chunk
            use_semantic: Whether to use semantic chunking

        Returns:
            List of chunked Documents with metadata
        """
        text = document.page_content

        if not text or len(text.strip()) < self.config.min_tokens_per_chunk:
            return [document]

        try:
            # Try semantic chunking first
            if use_semantic and self.config.use_semantic:
                chunks = self.semantic_splitter.split_documents([document])
            else:
                chunks = self.fallback_splitter.split_documents([document])

        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            chunks = self.fallback_splitter.split_documents([document])

        # Post-process chunks
        processed_chunks = self._post_process_chunks(chunks, document)

        return processed_chunks

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Chunk raw text.

        Args:
            text: Text to chunk
            document_id: Parent document identifier
            metadata: Additional metadata to attach

        Returns:
            List of chunked Documents
        """
        base_metadata = metadata or {}
        base_metadata["parent_document_id"] = document_id

        document = Document(page_content=text, metadata=base_metadata)
        return self.chunk_document(document)

    def _post_process_chunks(
        self,
        chunks: List[Document],
        parent_document: Document
    ) -> List[Document]:
        """
        Post-process chunks with metadata and validation.
        """
        parent_id = parent_document.metadata.get(
            "document_id",
            hashlib.md5(parent_document.page_content[:500].encode()).hexdigest()
        )

        processed = []
        char_offset = 0

        for i, chunk in enumerate(chunks):
            # Generate chunk ID
            chunk_id = hashlib.md5(
                f"{parent_id}:{i}:{chunk.page_content[:100]}".encode()
            ).hexdigest()[:16]

            # Count tokens
            token_count = self.count_tokens(chunk.page_content)

            # Skip chunks that are too small
            if token_count < self.config.min_tokens_per_chunk:
                continue

            # Split chunks that are too large
            if token_count > self.config.max_tokens_per_chunk:
                sub_chunks = self._split_large_chunk(chunk, parent_id, i)
                processed.extend(sub_chunks)
                continue

            # Find position in original text
            start_char = parent_document.page_content.find(
                chunk.page_content[:50],
                char_offset
            )
            if start_char == -1:
                start_char = char_offset
            end_char = start_char + len(chunk.page_content)
            char_offset = end_char

            # Enrich metadata
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_char": start_char,
                "end_char": end_char,
                "token_count": token_count,
                "parent_document_id": parent_id,
                "chunking_strategy": "semantic" if self.config.use_semantic else "recursive",
            })

            # Preserve parent metadata
            for key, value in parent_document.metadata.items():
                if key not in chunk.metadata:
                    chunk.metadata[key] = value

            processed.append(chunk)

        # Update total_chunks after filtering
        for chunk in processed:
            chunk.metadata["total_chunks"] = len(processed)

        return processed

    def _split_large_chunk(
        self,
        chunk: Document,
        parent_id: str,
        base_index: int
    ) -> List[Document]:
        """Split a chunk that exceeds token limit."""
        sub_chunks = self.fallback_splitter.split_documents([chunk])

        result = []
        for j, sub_chunk in enumerate(sub_chunks):
            token_count = self.count_tokens(sub_chunk.page_content)

            sub_chunk.metadata.update({
                "chunk_id": hashlib.md5(
                    f"{parent_id}:{base_index}:{j}:{sub_chunk.page_content[:50]}".encode()
                ).hexdigest()[:16],
                "chunk_index": base_index,
                "sub_chunk_index": j,
                "token_count": token_count,
                "parent_document_id": parent_id,
                "chunking_strategy": "recursive_fallback",
            })
            result.append(sub_chunk)

        return result


class DocumentAwareChunker:
    """
    Document-structure-aware chunking.

    Respects document structure like headers, paragraphs,
    and sections when splitting.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # Markdown splitter for structured documents
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

        # Fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def chunk_markdown(self, text: str, document_id: str) -> List[Document]:
        """Chunk markdown-formatted text."""
        try:
            chunks = self.md_splitter.split_text(text)
            # Further split if needed
            final_chunks = self.text_splitter.split_documents(chunks)
            return self._add_metadata(final_chunks, document_id)
        except Exception:
            return self.chunk_text(text, document_id)

    def chunk_text(self, text: str, document_id: str) -> List[Document]:
        """Chunk plain text."""
        doc = Document(page_content=text, metadata={"document_id": document_id})
        chunks = self.text_splitter.split_documents([doc])
        return self._add_metadata(chunks, document_id)

    def _add_metadata(
        self,
        chunks: List[Document],
        document_id: str
    ) -> List[Document]:
        """Add chunk metadata."""
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": hashlib.md5(
                    f"{document_id}:{i}".encode()
                ).hexdigest()[:16],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_document_id": document_id,
            })
        return chunks


# Export
__all__ = [
    'SemanticChunker',
    'DocumentAwareChunker',
    'ChunkingConfig',
    'ChunkMetadata',
]
