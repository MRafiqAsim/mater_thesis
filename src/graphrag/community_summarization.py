"""
Community Summarization Module
==============================
Generate LLM summaries for detected communities.

Features:
- Community content aggregation
- GPT-4o summary generation
- Hierarchical summarization (fine → coarse)
- Summary indexing for retrieval

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
class CommunitySummary:
    """Summary for a community."""
    community_id: str
    level: int
    summary: str
    key_entities: List[str]
    key_themes: List[str]
    member_count: int
    source_chunks: List[str]


@dataclass
class SummarizationConfig:
    """Configuration for community summarization."""
    model_deployment: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 800

    # Content settings
    max_entities_in_prompt: int = 30
    max_relationships_in_prompt: int = 50
    max_context_length: int = 10000


class CommunitySummarizer:
    """
    Generate summaries for communities using GPT-4o.

    Usage:
        summarizer = CommunitySummarizer(azure_endpoint, api_key)
        summary = summarizer.summarize_community(community, entities, relationships)
    """

    SYSTEM_PROMPT = get_prompt("gold", "graphrag_community_summarization", "system_prompt")

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[SummarizationConfig] = None
    ):
        """
        Initialize community summarizer.

        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            config: Summarization configuration
        """
        self.config = config or SummarizationConfig()

        from langchain_openai import AzureChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", get_prompt("gold", "graphrag_community_summarization", "user_prompt")),
        ])

        self.chain = self.prompt | self.llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def summarize_community(
        self,
        community_id: str,
        level: int,
        member_ids: List[str],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        chunk_ids: Optional[List[str]] = None
    ) -> CommunitySummary:
        """
        Generate summary for a community.

        Args:
            community_id: Community identifier
            level: Hierarchy level
            member_ids: List of entity IDs in community
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            chunk_ids: Source chunk IDs

        Returns:
            CommunitySummary
        """
        # Filter entities to community members
        member_set = set(member_ids)
        community_entities = [e for e in entities if e.get("id") in member_set]
        community_relationships = [
            r for r in relationships
            if r.get("source_id") in member_set or r.get("target_id") in member_set
        ]

        # Format entities
        entities_text = self._format_entities(
            community_entities[:self.config.max_entities_in_prompt]
        )

        # Format relationships
        relationships_text = self._format_relationships(
            community_relationships[:self.config.max_relationships_in_prompt]
        )

        # Generate summary
        response = self.chain.invoke({
            "community_id": community_id,
            "level": level,
            "member_count": len(member_ids),
            "entities": entities_text,
            "entity_count": len(community_entities),
            "relationships": relationships_text,
            "relationship_count": len(community_relationships),
        })

        # Parse response
        summary_text, key_themes = self._parse_response(response.content)

        # Extract key entities (top by mention count)
        key_entities = [
            e.get("name", e.get("id"))
            for e in sorted(
                community_entities,
                key=lambda x: x.get("mention_count", 0),
                reverse=True
            )[:5]
        ]

        return CommunitySummary(
            community_id=community_id,
            level=level,
            summary=summary_text,
            key_entities=key_entities,
            key_themes=key_themes,
            member_count=len(member_ids),
            source_chunks=chunk_ids or [],
        )

    def _format_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities for prompt."""
        lines = []
        for e in entities:
            name = e.get("name", "Unknown")
            etype = e.get("type", "UNKNOWN")
            desc = e.get("description", "")[:100]
            mentions = e.get("mention_count", 1)
            lines.append(f"- {name} ({etype}): {desc} [mentions: {mentions}]")
        return "\n".join(lines) if lines else "No entities"

    def _format_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Format relationships for prompt."""
        lines = []
        for r in relationships:
            source = r.get("source_name", r.get("source_id", "?"))
            target = r.get("target_name", r.get("target_id", "?"))
            rtype = r.get("type", "RELATED_TO")
            desc = r.get("description", "")[:50]
            lines.append(f"- {source} --[{rtype}]--> {target}: {desc}")
        return "\n".join(lines) if lines else "No relationships"

    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into summary and themes."""
        summary = response
        themes = []

        # Try to extract key themes
        if "Key themes:" in response:
            parts = response.split("Key themes:")
            summary = parts[0].strip()
            themes_text = parts[1].strip()

            # Parse themes (comma or newline separated)
            for line in themes_text.split("\n"):
                line = line.strip().strip("-").strip("•").strip()
                if line and len(line) < 100:
                    themes.append(line)

        return summary, themes[:5]

    def summarize_communities_batch(
        self,
        communities: List[Dict[str, Any]],
        all_entities: List[Dict[str, Any]],
        all_relationships: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[CommunitySummary]:
        """
        Summarize multiple communities.

        Args:
            communities: List of community dicts with 'id', 'level', 'members'
            all_entities: All entities
            all_relationships: All relationships
            progress_callback: Optional callback(current, total)

        Returns:
            List of CommunitySummary objects
        """
        summaries = []
        total = len(communities)

        for i, comm in enumerate(communities):
            try:
                summary = self.summarize_community(
                    community_id=comm["community_id"],
                    level=comm["level"],
                    member_ids=comm["members"],
                    entities=all_entities,
                    relationships=all_relationships,
                )
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to summarize community {comm['community_id']}: {e}")
                # Create placeholder summary
                summaries.append(CommunitySummary(
                    community_id=comm["community_id"],
                    level=comm["level"],
                    summary=f"Community of {len(comm['members'])} entities",
                    key_entities=[],
                    key_themes=[],
                    member_count=len(comm["members"]),
                    source_chunks=[],
                ))

            if progress_callback:
                progress_callback(i + 1, total)

        return summaries


class HierarchicalSummarizer:
    """
    Generate hierarchical summaries (fine levels summarized into coarse).
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        config: Optional[SummarizationConfig] = None
    ):
        self.config = config or SummarizationConfig()
        self.base_summarizer = CommunitySummarizer(azure_endpoint, api_key, config)

        from langchain_openai import AzureChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01",
            azure_deployment=self.config.model_deployment,
            temperature=self.config.temperature,
        )

        self.rollup_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("gold", "hierarchical_summarization", "system_prompt")),
            ("human", get_prompt("gold", "hierarchical_summarization", "user_prompt")),
        ])

    def summarize_hierarchy(
        self,
        hierarchy_summaries: Dict[int, List[CommunitySummary]],
    ) -> Dict[int, List[CommunitySummary]]:
        """
        Generate rollup summaries for higher levels.

        For each coarse-level community, combines child summaries.
        """
        # Already have level summaries, could do additional rollup
        # For now, return as-is (could implement child->parent rollup)
        return hierarchy_summaries


class CommunitySummaryIndexer:
    """
    Index community summaries for retrieval.
    """

    def __init__(
        self,
        search_client,
        embeddings,
        index_name: str = "community-summaries"
    ):
        self.search_client = search_client
        self.embeddings = embeddings
        self.index_name = index_name

    def index_summaries(
        self,
        summaries: List[CommunitySummary]
    ) -> tuple:
        """
        Index community summaries for search.

        Returns:
            Tuple of (success_count, error_count)
        """
        success = 0
        errors = 0

        for summary in summaries:
            try:
                # Generate embedding
                embedding = self.embeddings.embed_query(summary.summary)

                doc = {
                    "id": summary.community_id,
                    "community_id": summary.community_id,
                    "level": summary.level,
                    "summary": summary.summary,
                    "summary_vector": embedding,
                    "key_entities": summary.key_entities,
                    "key_themes": summary.key_themes,
                    "member_count": summary.member_count,
                }

                self.search_client.upload_documents([doc])
                success += 1

            except Exception as e:
                logger.warning(f"Failed to index summary {summary.community_id}: {e}")
                errors += 1

        return success, errors


# Export
__all__ = [
    'CommunitySummarizer',
    'HierarchicalSummarizer',
    'CommunitySummaryIndexer',
    'CommunitySummary',
    'SummarizationConfig',
]
