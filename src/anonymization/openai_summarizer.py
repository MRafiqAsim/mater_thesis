"""
OpenAI-based Text Summarizer

Uses OpenAI API for intelligent text summarization.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result of text summarization"""
    summary: str
    key_entities: List[str]
    key_topics: List[str]
    original_length: int
    summary_length: int
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "key_entities": self.key_entities,
            "key_topics": self.key_topics,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": self.compression_ratio,
        }


SUMMARIZATION_SYSTEM_PROMPT = """You are an expert text summarizer for a knowledge management system.
Your task is to create concise, accurate summaries that preserve key information.

Guidelines:
1. Capture the main points and essential information
2. Preserve important entities (people, organizations, dates, locations)
3. Maintain factual accuracy - never add information not in the source
4. Use clear, professional language
5. For technical content, preserve key terminology
6. Note any temporal information (dates, deadlines, versions)"""


SUMMARIZATION_USER_PROMPT_CONCISE = """Summarize the following text in a concise paragraph (max {max_length} characters).
Focus on the most important information.

Text:
\"\"\"
{text}
\"\"\"

Provide your response as JSON:
{{
  "summary": "your concise summary here",
  "key_entities": ["list", "of", "important", "entities"],
  "key_topics": ["main", "topics", "covered"]
}}"""


SUMMARIZATION_USER_PROMPT_DETAILED = """Provide a detailed summary of the following text (max {max_length} characters).
Include all significant points and supporting details.

Text:
\"\"\"
{text}
\"\"\"

Provide your response as JSON:
{{
  "summary": "your detailed summary here",
  "key_entities": ["list", "of", "important", "entities"],
  "key_topics": ["main", "topics", "covered"]
}}"""


SUMMARIZATION_USER_PROMPT_BULLETS = """Summarize the following text as bullet points (max {max_length} characters total).
Each bullet should capture a key point.

Text:
\"\"\"
{text}
\"\"\"

Provide your response as JSON:
{{
  "summary": "• Point 1\\n• Point 2\\n• Point 3",
  "key_entities": ["list", "of", "important", "entities"],
  "key_topics": ["main", "topics", "covered"]
}}"""


class OpenAISummarizer:
    """
    Text Summarizer using OpenAI API.

    Provides intelligent summarization with entity extraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_summary_length: int = 500
    ):
        """
        Initialize OpenAI Summarizer.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o)
            temperature: Temperature for generation
            max_summary_length: Maximum summary length in characters
        """
        self.model = model
        self.temperature = temperature
        self.max_summary_length = max_summary_length

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI Summarizer initialized with model: {model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def summarize(
        self,
        text: str,
        style: str = "concise",
        max_length: Optional[int] = None
    ) -> SummaryResult:
        """
        Summarize text.

        Args:
            text: Text to summarize
            style: "concise", "detailed", or "bullet_points"
            max_length: Override default max length

        Returns:
            SummaryResult with summary and metadata
        """
        if not text or not text.strip():
            return SummaryResult(
                summary="",
                key_entities=[],
                key_topics=[],
                original_length=0,
                summary_length=0,
                compression_ratio=1.0
            )

        max_length = max_length or self.max_summary_length

        # Select prompt based on style
        if style == "detailed":
            user_prompt = SUMMARIZATION_USER_PROMPT_DETAILED
        elif style == "bullet_points":
            user_prompt = SUMMARIZATION_USER_PROMPT_BULLETS
        else:
            user_prompt = SUMMARIZATION_USER_PROMPT_CONCISE

        try:
            import json

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt.format(
                        text=text,
                        max_length=max_length
                    )}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            summary = result.get("summary", "")
            key_entities = result.get("key_entities", [])
            key_topics = result.get("key_topics", [])

            return SummaryResult(
                summary=summary,
                key_entities=key_entities,
                key_topics=key_topics,
                original_length=len(text),
                summary_length=len(summary),
                compression_ratio=len(summary) / len(text) if text else 1.0
            )

        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            # Return truncated text as fallback
            fallback = text[:max_length] + "..." if len(text) > max_length else text
            return SummaryResult(
                summary=fallback,
                key_entities=[],
                key_topics=[],
                original_length=len(text),
                summary_length=len(fallback),
                compression_ratio=len(fallback) / len(text) if text else 1.0
            )

    def summarize_batch(
        self,
        texts: List[str],
        style: str = "concise"
    ) -> List[SummaryResult]:
        """
        Summarize multiple texts.

        Args:
            texts: List of texts to summarize
            style: Summary style

        Returns:
            List of SummaryResult objects
        """
        return [self.summarize(text, style) for text in texts]

    def extract_entities_and_relations(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Extract entities and their relationships from text.

        Useful for knowledge graph construction.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with entities and relations
        """
        import json

        prompt = """Analyze the following text and extract:
1. Named entities (people, organizations, locations, dates, concepts)
2. Relationships between entities
3. Key facts and claims

Text:
\"\"\"
{text}
\"\"\"

Return as JSON:
{{
  "entities": [
    {{"name": "entity name", "type": "PERSON|ORG|LOCATION|DATE|CONCEPT", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "entity1", "relation": "relation type", "target": "entity2", "context": "supporting text"}}
  ],
  "facts": [
    {{"subject": "entity", "predicate": "property", "object": "value", "confidence": 0.9}}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt.format(text=text)}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": [], "facts": []}
