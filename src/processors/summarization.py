"""
Summarization Module
====================
LLM-based summarization using Azure OpenAI GPT-4o.

Strategies:
1. Chunk summarization - Summarize individual chunks
2. Document summarization - Hierarchical (chunks → sections → document)
3. Email thread summarization - Conversation-aware summaries
4. Map-reduce - For very long documents

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from prompt_loader import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result from summarization."""
    summary: str
    source_text_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    tokens_used: int


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""
    # Model settings
    model_deployment: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 500

    # Summary settings
    target_length: str = "concise"  # concise, detailed, comprehensive
    preserve_entities: bool = True
    include_key_facts: bool = True

    # Language
    output_language: str = "same"  # same, en, nl

    # Retry settings
    max_retries: int = 3


class Summarizer:
    """
    LLM-based summarization using Azure OpenAI.

    Usage:
        summarizer = Summarizer(azure_endpoint, api_key)
        result = summarizer.summarize_text("Long document text...")
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[SummarizationConfig] = None
    ):
        """
        Initialize summarizer.

        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            config: Summarization configuration
        """
        self.config = config or SummarizationConfig()

        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Token counter
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def summarize_text(
        self,
        text: str,
        context: Optional[str] = None,
        language: Optional[str] = None
    ) -> SummaryResult:
        """
        Summarize a text passage.

        Args:
            text: Text to summarize
            context: Additional context (document title, type, etc.)
            language: Target language for summary

        Returns:
            SummaryResult
        """
        from langchain_core.prompts import ChatPromptTemplate

        # Build prompt based on settings
        length_instruction = {
            "concise": "Write a concise summary in 2-3 sentences.",
            "detailed": "Write a detailed summary covering main points in 4-6 sentences.",
            "comprehensive": "Write a comprehensive summary covering all key information.",
        }.get(self.config.target_length, "Write a concise summary.")

        language_instruction = ""
        if language and language != "same":
            lang_name = {"en": "English", "nl": "Dutch"}.get(language, language)
            language_instruction = f"Write the summary in {lang_name}."

        entity_instruction = ""
        if self.config.preserve_entities:
            entity_instruction = "Preserve important names, dates, and specific references."

        prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("silver", "document_summarization", "system_prompt")),
            ("human", get_prompt("silver", "document_summarization", "user_prompt")),
        ])

        context_section = f"Context: {context}\n" if context else ""

        # Invoke LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "text": text[:15000],  # Limit input size
            "context_section": context_section,
            "length_instruction": length_instruction,
            "language_instruction": language_instruction,
            "entity_instruction": entity_instruction,
        })

        summary = response.content.strip()
        source_tokens = self.count_tokens(text)
        summary_tokens = self.count_tokens(summary)

        return SummaryResult(
            summary=summary,
            source_text_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 0,
            model_used=self.config.model_deployment,
            tokens_used=source_tokens + summary_tokens,
        )

    def summarize_chunks(
        self,
        chunks: List[str],
        document_title: Optional[str] = None
    ) -> List[SummaryResult]:
        """
        Summarize multiple chunks.

        Args:
            chunks: List of text chunks
            document_title: Parent document title for context

        Returns:
            List of SummaryResults
        """
        results = []
        for i, chunk in enumerate(chunks):
            context = f"Chunk {i + 1} of {len(chunks)}"
            if document_title:
                context = f"{document_title} - {context}"

            result = self.summarize_text(chunk, context)
            results.append(result)

        return results

    def summarize_document_hierarchical(
        self,
        chunks: List[str],
        document_title: Optional[str] = None
    ) -> SummaryResult:
        """
        Hierarchical document summarization.

        1. Summarize each chunk
        2. Combine chunk summaries
        3. Create final document summary

        Args:
            chunks: Document chunks
            document_title: Document title

        Returns:
            Final document summary
        """
        # Step 1: Summarize chunks
        chunk_summaries = self.summarize_chunks(chunks, document_title)

        # Step 2: Combine summaries
        combined = "\n\n".join([r.summary for r in chunk_summaries])

        # Step 3: If combined is short enough, summarize directly
        if self.count_tokens(combined) <= 4000:
            return self.summarize_text(
                combined,
                context=f"Combined summaries from: {document_title}" if document_title else None,
            )

        # Step 4: If still too long, do another round of summarization
        # Split combined into groups
        group_size = 5
        groups = [
            chunk_summaries[i:i + group_size]
            for i in range(0, len(chunk_summaries), group_size)
        ]

        group_summaries = []
        for group in groups:
            group_text = "\n\n".join([r.summary for r in group])
            group_summary = self.summarize_text(group_text)
            group_summaries.append(group_summary.summary)

        # Final summary
        final_text = "\n\n".join(group_summaries)
        return self.summarize_text(
            final_text,
            context=f"Document summary: {document_title}" if document_title else None,
        )


class EmailThreadSummarizer:
    """
    Summarize email conversation threads.

    Understands email structure and conversation flow.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[SummarizationConfig] = None
    ):
        self.config = config or SummarizationConfig()

        from langchain_openai import AzureChatOpenAI

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def summarize_thread(
        self,
        emails: List[Dict[str, Any]],
    ) -> SummaryResult:
        """
        Summarize an email thread.

        Args:
            emails: List of email dictionaries with keys:
                   - subject, sender, date, body

        Returns:
            Thread summary
        """
        from langchain_core.prompts import ChatPromptTemplate

        # Format thread
        thread_text = self._format_thread(emails)

        prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("silver", "email_thread_summarization", "system_prompt")),
            ("human", get_prompt("silver", "email_thread_summarization", "user_prompt")),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"thread": thread_text[:15000]})

        return SummaryResult(
            summary=response.content.strip(),
            source_text_length=len(thread_text),
            summary_length=len(response.content),
            compression_ratio=len(response.content) / len(thread_text) if thread_text else 0,
            model_used=self.config.model_deployment,
            tokens_used=0,  # Approximate
        )

    def _format_thread(self, emails: List[Dict[str, Any]]) -> str:
        """Format email thread for summarization."""
        parts = []
        for i, email in enumerate(emails, 1):
            parts.append(f"""
--- Email {i} ---
From: {email.get('sender', 'Unknown')}
Date: {email.get('date', 'Unknown')}
Subject: {email.get('subject', '(No Subject)')}

{email.get('body', '')}
""")
        return "\n".join(parts)


class MapReduceSummarizer:
    """
    Map-Reduce summarization for very large documents.

    Suitable for documents that exceed token limits.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[SummarizationConfig] = None
    ):
        self.config = config or SummarizationConfig()
        self.base_summarizer = Summarizer(azure_endpoint, api_key, config)

    def summarize(
        self,
        text: str,
        chunk_size: int = 4000,
        document_title: Optional[str] = None
    ) -> SummaryResult:
        """
        Map-Reduce summarization.

        Args:
            text: Long document text
            chunk_size: Characters per chunk for map phase
            document_title: Document title

        Returns:
            Final summary
        """
        # Split into chunks
        chunks = self._split_text(text, chunk_size)

        if len(chunks) == 1:
            return self.base_summarizer.summarize_text(text, document_title)

        # Map phase: summarize each chunk
        map_results = []
        for i, chunk in enumerate(chunks):
            result = self.base_summarizer.summarize_text(
                chunk,
                context=f"Part {i + 1} of {len(chunks)}: {document_title}" if document_title else None
            )
            map_results.append(result.summary)

        # Reduce phase: combine summaries
        combined = "\n\n".join(map_results)

        # Recursive reduce if needed
        while self.base_summarizer.count_tokens(combined) > 4000:
            chunks = self._split_text(combined, chunk_size // 2)
            if len(chunks) == 1:
                break

            reduce_results = []
            for chunk in chunks:
                result = self.base_summarizer.summarize_text(chunk)
                reduce_results.append(result.summary)
            combined = "\n\n".join(reduce_results)

        # Final summary
        return self.base_summarizer.summarize_text(
            combined,
            context=f"Final summary: {document_title}" if document_title else None
        )

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks


# Export
__all__ = [
    'Summarizer',
    'EmailThreadSummarizer',
    'MapReduceSummarizer',
    'SummaryResult',
    'SummarizationConfig',
]
