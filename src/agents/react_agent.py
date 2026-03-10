"""
ReAct Agent Module
==================
Tool-augmented reasoning agent using LangGraph for multi-hop question answering.

Features:
- ReAct (Reasoning + Acting) loop
- LangGraph state machine
- Multi-hop reasoning with tool use
- Citation tracking
- Configurable reasoning depth

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Annotated, TypedDict, Sequence
from dataclasses import dataclass, field
import logging
import operator
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from prompt_loader import get_prompt

logger = logging.getLogger(__name__)


# ============================================
# Agent State
# ============================================

class AgentState(TypedDict):
    """State for the ReAct agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_step: int
    max_steps: int
    tools_used: List[str]
    sources_cited: List[Dict[str, Any]]
    final_answer: Optional[str]
    reasoning_trace: List[str]


# ============================================
# Configuration
# ============================================

@dataclass
class ReActConfig:
    """Configuration for ReAct agent."""
    # Model settings
    model_deployment: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000

    # Reasoning settings
    max_reasoning_steps: int = 6
    require_citations: bool = True

    # Tool settings
    max_tool_calls_per_step: int = 3

    # Response settings
    include_reasoning_trace: bool = False


# ============================================
# System Prompts
# ============================================

REACT_SYSTEM_PROMPT = get_prompt("retrieval", "react_agent_langgraph", "system_prompt")


# ============================================
# ReAct Agent
# ============================================

class ReActAgent:
    """
    ReAct agent for tool-augmented reasoning.

    Usage:
        agent = ReActAgent(llm, tools, config)
        response = agent.invoke("What projects is John working on?")
    """

    def __init__(
        self,
        llm,
        tools: List[BaseTool],
        config: Optional[ReActConfig] = None
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: Language model (Azure OpenAI)
            tools: List of agent tools
            config: Agent configuration
        """
        self.llm = llm
        self.tools = tools
        self.config = config or ReActConfig()
        self.tool_map = {tool.name: tool for tool in tools}

        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"- **{tool.name}**: {tool.description}"
            for tool in tools
        ])

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REACT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.tool_descriptions = tool_descriptions

        # Bind tools to LLM
        self.llm_with_tools = llm.bind_tools(tools)

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query using ReAct reasoning.

        Args:
            query: User question

        Returns:
            Dictionary with answer, sources, and reasoning trace
        """
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_step=0,
            max_steps=self.config.max_reasoning_steps,
            tools_used=[],
            sources_cited=[],
            final_answer=None,
            reasoning_trace=[]
        )

        # Run reasoning loop
        while state["current_step"] < state["max_steps"]:
            state = self._reasoning_step(state)

            # Check if we have a final answer
            if state["final_answer"]:
                break

            state["current_step"] += 1

        # If no final answer after max steps, synthesize one
        if not state["final_answer"]:
            state = self._synthesize_answer(state)

        return {
            "answer": state["final_answer"],
            "sources": state["sources_cited"],
            "tools_used": state["tools_used"],
            "reasoning_trace": state["reasoning_trace"] if self.config.include_reasoning_trace else [],
            "steps": state["current_step"]
        }

    def _reasoning_step(self, state: AgentState) -> AgentState:
        """Execute one reasoning step."""
        # Build messages with system prompt
        messages = self.prompt.format_messages(
            tool_descriptions=self.tool_descriptions,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            messages=state["messages"]
        )

        # Get LLM response
        response = self.llm_with_tools.invoke(messages)

        # Add AI message to state
        state["messages"] = list(state["messages"]) + [response]

        # Check for tool calls
        if response.tool_calls:
            # Execute tools
            tool_messages = []

            for tool_call in response.tool_calls[:self.config.max_tool_calls_per_step]:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                state["tools_used"].append(tool_name)
                state["reasoning_trace"].append(
                    f"Step {state['current_step']}: Using {tool_name} with {tool_args}"
                )

                try:
                    tool = self.tool_map.get(tool_name)
                    if tool:
                        result = tool.invoke(tool_args)

                        # Track sources
                        self._extract_sources(tool_name, tool_args, result, state)

                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    else:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Tool '{tool_name}' not found",
                                tool_call_id=tool_call["id"]
                            )
                        )
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                    )

            # Add tool messages
            state["messages"] = list(state["messages"]) + tool_messages

        else:
            # No tool calls - this is the final answer
            state["final_answer"] = response.content
            state["reasoning_trace"].append(
                f"Step {state['current_step']}: Generated final answer"
            )

        return state

    def _extract_sources(
        self,
        tool_name: str,
        tool_args: Dict,
        result: str,
        state: AgentState
    ):
        """Extract source citations from tool results."""
        if tool_name == "vector_search":
            # Extract document sources from vector search results
            if "Source:" in result:
                for line in result.split("\n"):
                    if line.startswith("Source:"):
                        source = line.replace("Source:", "").strip()
                        state["sources_cited"].append({
                            "type": "document",
                            "source": source,
                            "tool": tool_name
                        })

        elif tool_name == "entity_lookup":
            entity_name = tool_args.get("entity_name", "")
            state["sources_cited"].append({
                "type": "entity",
                "source": entity_name,
                "tool": tool_name
            })

        elif tool_name == "community_search":
            if "Community" in result:
                for line in result.split("\n"):
                    if "**Community" in line:
                        # Extract community ID
                        start = line.find("Community ") + 10
                        end = line.find("**", start)
                        if end > start:
                            comm_id = line[start:end]
                            state["sources_cited"].append({
                                "type": "community",
                                "source": comm_id,
                                "tool": tool_name
                            })

    def _synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from gathered information."""
        synthesis_prompt = get_prompt("retrieval", "react_agent_langgraph", "synthesis_user_prompt")

        state["messages"] = list(state["messages"]) + [
            HumanMessage(content=synthesis_prompt)
        ]

        # Get final response (without tools)
        messages = self.prompt.format_messages(
            tool_descriptions=self.tool_descriptions,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            messages=state["messages"]
        )

        response = self.llm.invoke(messages)
        state["final_answer"] = response.content

        return state


# ============================================
# LangGraph Implementation
# ============================================

def create_react_graph(
    llm,
    tools: List[BaseTool],
    config: Optional[ReActConfig] = None
):
    """
    Create a LangGraph-based ReAct agent.

    Args:
        llm: Language model
        tools: List of tools
        config: Agent configuration

    Returns:
        Compiled LangGraph
    """
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode

    config = config or ReActConfig()
    tool_map = {tool.name: tool for tool in tools}

    # Build tool descriptions
    tool_descriptions = "\n".join([
        f"- **{tool.name}**: {tool.description}"
        for tool in tools
    ])

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    def should_continue(state: AgentState) -> str:
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If LLM made tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Check step limit
        if state["current_step"] >= state["max_steps"]:
            return "end"

        return "end"

    def call_model(state: AgentState) -> AgentState:
        """Call the model with current state."""
        messages = prompt.format_messages(
            tool_descriptions=tool_descriptions,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            messages=state["messages"]
        )

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "current_step": state["current_step"] + 1,
            "tools_used": state["tools_used"],
            "sources_cited": state["sources_cited"],
            "final_answer": response.content if not (hasattr(response, "tool_calls") and response.tool_calls) else None,
            "reasoning_trace": state["reasoning_trace"] + [f"Step {state['current_step']}: Called model"],
            "max_steps": state["max_steps"]
        }

    def call_tools(state: AgentState) -> AgentState:
        """Execute tool calls."""
        messages = state["messages"]
        last_message = messages[-1]

        tool_messages = []
        tools_used = list(state["tools_used"])

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tools_used.append(tool_name)

            try:
                tool = tool_map.get(tool_name)
                if tool:
                    result = tool.invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool '{tool_name}' not found",
                            tool_call_id=tool_call["id"]
                        )
                    )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool error: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                )

        return {
            "messages": tool_messages,
            "current_step": state["current_step"],
            "tools_used": tools_used,
            "sources_cited": state["sources_cited"],
            "final_answer": None,
            "reasoning_trace": state["reasoning_trace"],
            "max_steps": state["max_steps"]
        }

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Tools always go back to agent
    workflow.add_edge("tools", "agent")

    # Compile
    return workflow.compile()


# ============================================
# Multi-Hop QA Agent
# ============================================

class MultiHopQAAgent:
    """
    Specialized agent for multi-hop question answering.

    Decomposes complex questions into sub-questions and
    aggregates answers.
    """

    SYSTEM_PROMPT = get_prompt("retrieval", "multi_hop_qa", "system_prompt")
    DECOMPOSITION_PROMPT = get_prompt("retrieval", "multi_hop_qa", "decomposition_user_prompt")
    SYNTHESIS_PROMPT = get_prompt("retrieval", "multi_hop_qa", "synthesis_user_prompt")

    def __init__(
        self,
        llm,
        tools: List[BaseTool],
        config: Optional[ReActConfig] = None
    ):
        self.llm = llm
        self.react_agent = ReActAgent(llm, tools, config)
        self.config = config or ReActConfig()

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Answer a complex question using multi-hop reasoning.

        Args:
            question: Complex question requiring multiple reasoning steps

        Returns:
            Dictionary with answer, sub-answers, and sources
        """
        # Step 1: Decompose question
        sub_questions = self._decompose_question(question)

        logger.info(f"Decomposed into {len(sub_questions)} sub-questions")

        # Step 2: Answer each sub-question
        sub_answers = []
        all_sources = []

        for i, sub_q in enumerate(sub_questions):
            logger.info(f"Answering sub-question {i+1}: {sub_q[:50]}...")

            result = self.react_agent.invoke(sub_q)

            sub_answers.append({
                "question": sub_q,
                "answer": result["answer"],
                "sources": result["sources"]
            })

            all_sources.extend(result["sources"])

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answer(question, sub_answers)

        return {
            "answer": final_answer,
            "sub_answers": sub_answers,
            "sources": all_sources,
            "num_hops": len(sub_questions)
        }

    def _decompose_question(self, question: str) -> List[str]:
        """Decompose complex question into sub-questions."""
        from langchain_core.messages import SystemMessage, HumanMessage as HMsg
        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HMsg(content=self.DECOMPOSITION_PROMPT.format(question=question))
        ])

        # Parse numbered list
        lines = response.content.strip().split("\n")
        sub_questions = []

        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                sub_q = line.split(".", 1)[-1].strip()
                if sub_q:
                    sub_questions.append(sub_q)

        # Fallback: if decomposition failed, use original
        if not sub_questions:
            sub_questions = [question]

        return sub_questions[:4]  # Limit to 4 sub-questions

    def _synthesize_answer(
        self,
        question: str,
        sub_answers: List[Dict]
    ) -> str:
        """Synthesize final answer from sub-answers."""
        # Format sub-QA pairs
        sub_qa_text = ""
        for i, sa in enumerate(sub_answers, 1):
            sub_qa_text += f"\n{i}. Q: {sa['question']}\n   A: {sa['answer']}\n"

        from langchain_core.messages import SystemMessage, HumanMessage as HMsg
        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HMsg(content=self.SYNTHESIS_PROMPT.format(
                question=question,
                sub_qa_pairs=sub_qa_text
            ))
        ])

        return response.content


# Export
__all__ = [
    'ReActAgent',
    'MultiHopQAAgent',
    'ReActConfig',
    'AgentState',
    'create_react_graph',
]
