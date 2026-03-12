"""
ReAct Retriever Module

Implements a ReAct (Reasoning + Acting) agent for intelligent retrieval
that combines PathRAG, GraphRAG, and vector search strategies.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    from prompt_loader import get_prompt, format_prompt
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prompt_loader import get_prompt, format_prompt

from .retrieval_tools import RetrievalToolkit, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning chain."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "is_final": self.is_final
        }


@dataclass
class ReActResult:
    """Result of ReAct agent execution."""
    query: str
    answer: str
    steps: List[ReActStep]
    sources: List[Dict[str, Any]]
    total_tokens: int = 0
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "sources": self.sources,
            "total_tokens": self.total_tokens,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class ReActConfig:
    """Configuration for ReAct agent."""
    max_steps: int = 10
    temperature: float = 0.0
    model: str = "gpt-4o"
    verbose: bool = True


class ReActRetriever:
    """
    ReAct (Reasoning + Acting) agent for intelligent retrieval.

    Uses iterative reasoning to:
    1. Analyze the query and plan retrieval strategy
    2. Execute appropriate tools (PathRAG, GraphRAG, Vector, etc.)
    3. Synthesize observations into a coherent answer
    4. Cite sources for transparency
    """

    def __init__(
        self,
        gold_path: str,
        silver_path: Optional[str] = None,
        config: Optional[ReActConfig] = None,
        mode: str = "local",
    ):
        """
        Initialize the ReAct retriever.

        Args:
            gold_path: Path to Gold layer with indexes
            silver_path: Path to Silver layer with chunks
            config: Agent configuration
            mode: Processing mode — "local" uses local embeddings
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None
        self.config = config or ReActConfig()

        # Initialize toolkit
        self.toolkit = RetrievalToolkit(str(gold_path), str(silver_path) if silver_path else None, mode=mode)

        # Initialize LLM client
        self.client = None
        self.use_azure = False
        self._initialize_client()

        logger.info("ReActRetriever initialized")

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")

            if azure_endpoint and azure_key:
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                self.use_azure = True
                self.config.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                logger.info("Using Azure OpenAI for ReAct")
                return

            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
                self.use_azure = False
                logger.info("Using OpenAI for ReAct")
                return

            logger.warning("No OpenAI credentials found")

        except ImportError:
            logger.error("openai package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")

    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for the system prompt."""
        descriptions = []
        for tool in self.toolkit.tools.values():
            params = ", ".join([
                f"{name}: {info.get('type', 'string')}"
                for name, info in tool.parameters.items()
                if info.get('required', False)
            ])
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)

    def _parse_response(self, response_text: str) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse LLM response to extract thought, action, action_input, or final_answer.

        Returns: (thought, action, action_input, final_answer)
        """
        thought = None
        action = None
        action_input = None
        final_answer = None

        lines = response_text.strip().split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith('THOUGHT:'):
                if current_section == 'thought':
                    thought = ' '.join(current_content).strip()
                current_section = 'thought'
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']

            elif line_upper.startswith('ACTION:'):
                if current_section == 'thought':
                    thought = ' '.join(current_content).strip()
                current_section = 'action'
                action = line.split(':', 1)[1].strip() if ':' in line else ''

            elif line_upper.startswith('ACTION_INPUT:'):
                current_section = 'action_input'
                input_text = line.split(':', 1)[1].strip() if ':' in line else ''
                current_content = [input_text]

            elif line_upper.startswith('FINAL_ANSWER:'):
                if current_section == 'thought':
                    thought = ' '.join(current_content).strip()
                current_section = 'final_answer'
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']

            elif current_section:
                current_content.append(line)

        # Process final section
        if current_section == 'thought':
            thought = ' '.join(current_content).strip()
        elif current_section == 'action_input':
            try:
                input_text = ' '.join(current_content).strip()
                action_input = json.loads(input_text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{[^}]+\}', input_text)
                if json_match:
                    try:
                        action_input = json.loads(json_match.group())
                    except:
                        action_input = {}
        elif current_section == 'final_answer':
            final_answer = '\n'.join(current_content).strip()

        return thought, action, action_input, final_answer

    def query(self, question: str) -> ReActResult:
        """
        Execute a query using the ReAct loop.

        Args:
            question: The user's question

        Returns:
            ReActResult with answer, steps, and sources
        """
        start_time = datetime.now()

        if not self.client:
            return ReActResult(
                query=question,
                answer="",
                steps=[],
                sources=[],
                success=False,
                error_message="LLM client not initialized"
            )

        # Build system prompt from config/prompts.json
        system_prompt = format_prompt(
            get_prompt("retrieval", "react_agent", "system_prompt", "You are a retrieval agent."),
            tool_descriptions=self._build_tool_descriptions()
        )

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {question}"}
        ]

        steps: List[ReActStep] = []
        sources: List[Dict[str, Any]] = []
        total_tokens = 0

        # ReAct loop
        for step_num in range(1, self.config.max_steps + 1):
            try:
                # Get LLM response
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=2000
                )

                response_text = response.choices[0].message.content
                total_tokens += response.usage.total_tokens if response.usage else 0

                if self.config.verbose:
                    logger.info(f"Step {step_num}: {response_text[:200]}...")

                # Parse response
                thought, action, action_input, final_answer = self._parse_response(response_text)

                # Create step
                step = ReActStep(
                    step_number=step_num,
                    thought=thought or "",
                    action=action,
                    action_input=action_input
                )

                # Check for final answer
                if final_answer:
                    step.is_final = True
                    steps.append(step)

                    execution_time = (datetime.now() - start_time).total_seconds()

                    return ReActResult(
                        query=question,
                        answer=final_answer,
                        steps=steps,
                        sources=sources,
                        total_tokens=total_tokens,
                        execution_time=execution_time,
                        success=True
                    )

                # Execute action
                if action and action_input is not None:
                    tool_result = self.toolkit.execute_tool(action, **action_input)
                    observation = self._format_observation(tool_result)
                    step.observation = observation

                    # Collect sources
                    if tool_result.success and tool_result.data:
                        sources.extend(self._extract_sources(tool_result))

                    # Add to conversation
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})

                steps.append(step)

            except Exception as e:
                logger.error(f"ReAct step {step_num} failed: {e}")
                steps.append(ReActStep(
                    step_number=step_num,
                    thought=f"Error: {str(e)}",
                    is_final=True
                ))
                break

        # If we reach here, we exceeded max steps
        execution_time = (datetime.now() - start_time).total_seconds()

        # Generate summary answer from collected information
        summary_answer = self._generate_summary_answer(question, steps, sources)

        return ReActResult(
            query=question,
            answer=summary_answer,
            steps=steps,
            sources=sources,
            total_tokens=total_tokens,
            execution_time=execution_time,
            success=True
        )

    def _format_observation(self, tool_result: ToolResult) -> str:
        """Format tool result as observation string."""
        if not tool_result.success:
            return f"Error: {tool_result.message}"

        data = tool_result.data
        if isinstance(data, list):
            if len(data) == 0:
                return "No results found."
            # For entity listings and type discovery, show more items
            is_entity_listing = tool_result.tool_name in ("list_entities", "list_entity_types")
            max_items = 20 if is_entity_listing else 5
            # Summarize list results
            summaries = []
            for item in data[:max_items]:
                if isinstance(item, dict):
                    # Format dict nicely — always include IDs so the agent can reference them
                    summary_parts = []
                    for key in ['chunk_id', 'community_id', 'path_id', 'type', 'count', 'description', 'summary', 'text', 'name', 'path_type', 'connections']:
                        if key in item:
                            value = str(item[key])[:200]
                            summary_parts.append(f"{key}: {value}")
                    summaries.append(" | ".join(summary_parts[:5]))
                else:
                    summaries.append(str(item)[:100])
            result_text = f"Found {len(data)} results:\n" + "\n".join(f"- {s}" for s in summaries)
            if len(data) > max_items:
                result_text += f"\n... and {len(data) - max_items} more"
            return result_text

        elif isinstance(data, dict):
            # For global_search / local_search, the answer is the primary output
            if "answer" in data and tool_result.tool_name in ("global_search", "local_search"):
                answer = data["answer"]
                meta_parts = []
                for k, v in data.items():
                    if k == "answer":
                        continue
                    if isinstance(v, list):
                        meta_parts.append(f"{k}: [{len(v)} items]")
                    elif isinstance(v, str) and len(v) > 100:
                        meta_parts.append(f"{k}: {v[:100]}...")
                    else:
                        meta_parts.append(f"{k}: {v}")
                meta = " | ".join(meta_parts)
                return f"Answer: {answer}\n\nMetadata: {meta}" if meta else f"Answer: {answer}"

            # Format single result
            parts = []
            for key, value in list(data.items())[:10]:
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                elif isinstance(value, list):
                    value = f"[{len(value)} items]"
                parts.append(f"{key}: {value}")
            return "\n".join(parts)

        return str(data)[:500]

    def _extract_sources(self, tool_result: ToolResult) -> List[Dict[str, Any]]:
        """Extract source references from tool result."""
        sources = []
        data = tool_result.data

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    source = {}
                    if 'chunk_id' in item:
                        source['chunk_id'] = item['chunk_id']
                        source['type'] = 'chunk'
                    elif 'community_id' in item:
                        source['community_id'] = item['community_id']
                        source['type'] = 'community'
                        # Promote community source chunks to individual chunk sources
                        for cid in item.get('source_chunk_ids', [])[:10]:
                            sources.append({'chunk_id': cid, 'type': 'chunk'})
                    elif 'path_id' in item:
                        source['path_id'] = item['path_id']
                        source['type'] = 'path'
                        # Promote evidence_chunks to individual chunk sources
                        for eid in item.get('evidence_chunks', [])[:5]:
                            sources.append({'chunk_id': eid, 'type': 'chunk'})
                    if source:
                        sources.append(source)

        elif isinstance(data, dict):
            if 'chunk_id' in data:
                sources.append({'chunk_id': data['chunk_id'], 'type': 'chunk'})
            if 'source_chunks' in data:
                for chunk_id in data['source_chunks'][:5]:
                    sources.append({'chunk_id': chunk_id, 'type': 'chunk'})
            # global_search / local_search return source_chunk_ids
            if 'source_chunk_ids' in data:
                for chunk_id in data['source_chunk_ids'][:20]:
                    sources.append({'chunk_id': chunk_id, 'type': 'chunk'})

        return sources

    def _generate_summary_answer(
        self,
        question: str,
        steps: List[ReActStep],
        sources: List[Dict[str, Any]]
    ) -> str:
        """Generate a synthesized answer when max steps reached without FINAL_ANSWER."""
        # Collect all observations
        observations = []
        for step in steps:
            if step.observation:
                observations.append(step.observation)

        if not observations:
            return "I was unable to find relevant information to answer this question."

        # Use LLM to synthesize a proper answer from the raw observations
        if self.client:
            context = "\n\n---\n\n".join(obs[:600] for obs in observations[:5])
            fallback_sys = get_prompt("retrieval", "react_agent", "fallback_system_prompt",
                "You are a helpful assistant that answers questions based ONLY on the provided context.")
            fallback_user = get_prompt("retrieval", "react_agent", "fallback_user_prompt",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer based ONLY on the context above.")
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": fallback_sys},
                        {"role": "user", "content": fallback_user.format(
                            context=context, question=question
                        )},
                    ],
                    temperature=get_prompt("retrieval", "react_agent", "temperature", 0.3),
                    max_tokens=get_prompt("retrieval", "react_agent", "max_tokens", 1000),
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Summary answer generation failed: {e}")

        # Last resort fallback if LLM call fails
        return "I found relevant information but was unable to synthesize a complete answer. Please try rephrasing your question."

    def is_available(self) -> bool:
        """Check if the retriever is ready to use."""
        return self.client is not None
