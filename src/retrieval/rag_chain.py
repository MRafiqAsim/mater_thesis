"""
RAG Chain Module
================
LangChain-based Retrieval-Augmented Generation pipeline.

Features:
- Azure AI Search retriever integration
- GPT-4o generation with citations
- Context compression
- Query transformation
- Answer grounding

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: str
    tokens_used: int
    retrieval_time_ms: float
    generation_time_ms: float


@dataclass
class RAGConfig:
    """Configuration for RAG chain."""
    # Retrieval settings
    top_k: int = 5
    search_type: str = "hybrid"  # vector, keyword, hybrid
    min_relevance_score: float = 0.5

    # Generation settings
    model_deployment: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 1000

    # Context settings
    max_context_length: int = 12000  # tokens
    include_metadata: bool = True

    # Prompt settings
    system_prompt: str = """You are a knowledgeable assistant that answers questions based on the provided context.

Guidelines:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite sources using [1], [2], etc. format
- Be concise but comprehensive
- Maintain factual accuracy"""


class AzureSearchRetriever:
    """
    LangChain-compatible retriever for Azure AI Search.
    """

    def __init__(
        self,
        searcher,  # HybridSearcher instance
        top_k: int = 5,
        search_type: str = "hybrid"
    ):
        self.searcher = searcher
        self.top_k = top_k
        self.search_type = search_type

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for query."""
        response = self.searcher.search(
            query,
            top_k=self.top_k,
            search_type=self.search_type
        )

        documents = []
        for result in response.results:
            doc = Document(
                page_content=result.content,
                metadata={
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "reranker_score": result.reranker_score,
                    **result.metadata,
                }
            )
            documents.append(doc)

        return documents

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)


class RAGChain:
    """
    Basic RAG chain with Azure AI Search and GPT-4o.

    Usage:
        rag = RAGChain(searcher, azure_endpoint, api_key)
        response = rag.query("What is the project timeline?")
    """

    def __init__(
        self,
        searcher,  # HybridSearcher instance
        azure_endpoint: str,
        api_key: str,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG chain.

        Args:
            searcher: HybridSearcher instance
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        self.searcher = searcher

        # Initialize retriever
        self.retriever = AzureSearchRetriever(
            searcher,
            top_k=self.config.top_k,
            search_type=self.config.search_type
        )

        # Initialize LLM
        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Build prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", """Context:
{context}

Question: {question}

Answer (cite sources using [1], [2], etc.):"""),
        ])

        # Build chain
        self.chain = (
            {
                "context": lambda x: self._format_context(
                    self.retriever.get_relevant_documents(x["question"])
                ),
                "question": lambda x: x["question"],
                "docs": lambda x: self.retriever.get_relevant_documents(x["question"]),
            }
            | RunnablePassthrough.assign(
                answer=self.prompt | self.llm | StrOutputParser()
            )
        )

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context string."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            # Build context entry
            content = doc.page_content

            if self.config.include_metadata:
                source = doc.metadata.get("source_file", "Unknown")
                doc_type = doc.metadata.get("parent_type", "document")
                context_parts.append(f"[{i}] Source: {source} ({doc_type})\n{content}")
            else:
                context_parts.append(f"[{i}] {content}")

        return "\n\n---\n\n".join(context_parts)

    def query(self, question: str) -> RAGResponse:
        """
        Execute RAG query.

        Args:
            question: User question

        Returns:
            RAGResponse with answer and sources
        """
        import time

        # Retrieval
        retrieval_start = time.time()
        docs = self.retriever.get_relevant_documents(question)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Format context
        context = self._format_context(docs)

        # Generation
        generation_start = time.time()
        response = self.chain.invoke({"question": question})
        generation_time_ms = (time.time() - generation_start) * 1000

        # Build sources
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "index": i,
                "chunk_id": doc.metadata.get("chunk_id"),
                "content_preview": doc.page_content[:200] + "...",
                "source_file": doc.metadata.get("source_file"),
                "score": doc.metadata.get("score"),
                "reranker_score": doc.metadata.get("reranker_score"),
            })

        return RAGResponse(
            answer=response["answer"],
            sources=sources,
            query=question,
            context_used=context,
            tokens_used=0,  # Could calculate from response
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )

    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Execute multiple queries."""
        return [self.query(q) for q in questions]


class QueryTransformer:
    """
    Transform queries for better retrieval.

    Techniques:
    - Query expansion
    - Hypothetical document generation (HyDE)
    - Multi-query generation
    """

    def __init__(self, azure_endpoint: str, api_key: str):
        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment="gpt-4o",
            temperature=0.7,
        )

    def expand_query(self, query: str) -> List[str]:
        """Generate query variations."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate 3 alternative phrasings of the given question to improve search retrieval. Return only the questions, one per line."),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})

        # Parse result
        variations = [query]  # Include original
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line not in variations:
                variations.append(line)

        return variations[:4]  # Limit to 4 total

    def generate_hyde(self, query: str) -> str:
        """
        Generate hypothetical document (HyDE technique).

        Creates a hypothetical answer that would be returned,
        then uses that for embedding-based search.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a question, write a short paragraph that would be the ideal answer found in a document. Be specific and factual."),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})


class ContextCompressor:
    """
    Compress and filter retrieved context.

    Techniques:
    - LLM-based relevance filtering
    - Extractive compression
    """

    def __init__(self, azure_endpoint: str, api_key: str):
        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment="gpt-4o",
            temperature=0,
        )

    def filter_relevant(
        self,
        documents: List[Document],
        query: str,
        max_docs: int = 5
    ) -> List[Document]:
        """Filter documents by relevance to query."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate the relevance of each document to the query.
Return a JSON array of objects with 'index' (1-based) and 'score' (0-10).
Only include documents with score >= 5."""),
            ("human", """Query: {query}

Documents:
{documents}

Return JSON:"""),
        ])

        # Format documents
        doc_text = "\n\n".join([
            f"[{i+1}] {doc.page_content[:500]}"
            for i, doc in enumerate(documents)
        ])

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "documents": doc_text})

        # Parse result
        try:
            import json
            scores = json.loads(result)
            relevant_indices = [s["index"] - 1 for s in scores if s.get("score", 0) >= 5]
            return [documents[i] for i in relevant_indices[:max_docs]]
        except Exception:
            # Fallback to top documents
            return documents[:max_docs]

    def compress_document(self, document: Document, query: str) -> Document:
        """Extract only relevant portions from document."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract only the sentences from the document that are relevant to answering the query. Return the extracted text only."),
            ("human", """Query: {query}

Document: {document}

Relevant excerpts:"""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        compressed = chain.invoke({
            "query": query,
            "document": document.page_content
        })

        return Document(
            page_content=compressed,
            metadata={**document.metadata, "compressed": True}
        )


# Export
__all__ = [
    'RAGChain',
    'RAGResponse',
    'RAGConfig',
    'AzureSearchRetriever',
    'QueryTransformer',
    'ContextCompressor',
]
