#!/usr/bin/env python3
"""
Retrieval Strategy Comparison — Gradio Web Interface

Interactive comparison of 5 retrieval strategies with multi-turn chat:
  Vector | PathRAG | GraphRAG | Hybrid | ReAct

Tabs:
  1. Chat       — Multi-turn conversational Q&A with follow-up support
  2. Single Query — One-shot query with detailed metadata
  3. Compare    — Side-by-side strategy comparison

Usage:
    python -m src.app --mode local
    python -m src.app --gold ./data/gold_local --silver ./data/silver_local
    python -m src.app --port 7861
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import tiktoken

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import gradio as gr

from src.retrieval import HybridRetriever, RetrievalStrategy, RetrievalResult
from src.prompt_loader import get_prompt, format_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton retriever
# ---------------------------------------------------------------------------
retriever: HybridRetriever | None = None
_gold_path: str = ""
_silver_path: str = ""
_mode: str = "local"
_cosmos_adapter = None

STRATEGY_LABELS = ["Vector", "PathRAG", "GraphRAG", "Hybrid", "ReAct"]
STRATEGY_MAP = {
    "Vector": RetrievalStrategy.VECTOR,
    "PathRAG": RetrievalStrategy.PATHRAG,
    "GraphRAG": RetrievalStrategy.GRAPHRAG,
    "Hybrid": RetrievalStrategy.HYBRID,
    "ReAct": RetrievalStrategy.REACT,
}

FALLBACK_EXAMPLES = [
    ["What projects were discussed in the emails and who was involved?"],
    ["What technical issues were reported and how were they resolved?"],
    ["Who are the main contacts and what roles do they play?"],
]

# ---------------------------------------------------------------------------
# Conversation context management — token-based compaction
# ---------------------------------------------------------------------------
_enc = tiktoken.encoding_for_model("gpt-4o")

MAX_CONTEXT_TOKENS = 128_000      # GPT-4o context window
RESPONSE_RESERVE = 4_000          # Tokens reserved for LLM response
COMPACTION_THRESHOLD = 0.90       # Trigger compaction at 90% of budget
CONTEXT_BUDGET = MAX_CONTEXT_TOKENS - RESPONSE_RESERVE
# Approximate fixed overhead per call (system prompt + chunk context + query)
FIXED_OVERHEAD_ESTIMATE = 6_000


def count_tokens(text: str) -> int:
    """Count tokens using GPT-4o tokenizer."""
    return len(_enc.encode(text))


def build_history_text(conv_state: List[Dict[str, str]]) -> str:
    """Build a formatted history string from conversation state."""
    lines = []
    summary = conv_state[0].get("_compaction_summary", "") if conv_state else ""
    if summary:
        lines.append(f"[Prior conversation summary]: {summary}")
    for turn in conv_state:
        if "_compaction_summary" in turn:
            continue
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def history_token_count(conv_state: List[Dict[str, str]]) -> int:
    """Count total tokens in conversation history."""
    return count_tokens(build_history_text(conv_state))


def compact_history(conv_state: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], bool]:
    """
    Gradually compact conversation history — only summarize the oldest batch
    of turns needed to get back under the token threshold.

    Strategy: summarize the oldest half of real turns, keep the rest verbatim.
    This preserves recent context while compressing older exchanges.
    Returns (new_conv_state, was_compacted).
    """
    real_turns = [t for t in conv_state if "_compaction_summary" not in t]
    if len(real_turns) < 4:
        return conv_state, False

    # Summarize the oldest half, keep the recent half verbatim
    split_point = len(real_turns) // 2
    older = real_turns[:split_point]
    recent = real_turns[split_point:]

    # Build text to summarize — include previous compaction summary for continuity
    older_text_parts = []
    if conv_state and "_compaction_summary" in conv_state[0]:
        older_text_parts.append(f"Previous summary: {conv_state[0]['_compaction_summary']}")
    for turn in older:
        older_text_parts.append(f"User: {turn['user']}")
        older_text_parts.append(f"Assistant: {turn['answer']}")
    older_text = "\n".join(older_text_parts)

    # Scale summary length to the amount of content being compressed
    older_tokens = count_tokens(older_text)
    # Target: ~25% of original size, capped between 200-2000 tokens
    target_summary_tokens = max(200, min(2000, older_tokens // 4))

    r = get_retriever()
    if not r.llm_client:
        summary_text = f"Conversation covered {len(older)} earlier exchanges."
        return [{"_compaction_summary": summary_text}] + recent, True

    try:
        response = r.llm_client.chat.completions.create(
            model=r.config.answer_model,
            messages=[
                {"role": "system", "content": (
                    "Summarize this conversation history. Preserve all key topics, "
                    "entity names (PERSON_001, ORG_003 etc.), specific findings, and "
                    "conclusions discussed. Keep enough detail that a follow-up question "
                    "about any topic mentioned can still be answered accurately."
                )},
                {"role": "user", "content": older_text},
            ],
            temperature=0.0,
            max_tokens=target_summary_tokens,
        )
        summary_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Compaction summary failed: {e}")
        summary_text = f"Conversation covered {len(older)} earlier exchanges."

    new_state = [{"_compaction_summary": summary_text}] + recent
    summary_tokens = count_tokens(summary_text)
    logger.info(
        f"Compacted {len(older)} turns ({older_tokens} tokens) into summary "
        f"({summary_tokens} tokens), kept {len(recent)} recent turns verbatim"
    )
    return new_state, True


def maybe_compact(conv_state: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], bool]:
    """Check token budget and compact if needed. Returns (state, was_compacted)."""
    hist_tokens = history_token_count(conv_state)
    available = CONTEXT_BUDGET - FIXED_OVERHEAD_ESTIMATE
    threshold = available * COMPACTION_THRESHOLD

    if hist_tokens > threshold:
        logger.info(f"History tokens ({hist_tokens}) exceed {COMPACTION_THRESHOLD:.0%} of budget ({threshold:.0f}). Compacting...")
        return compact_history(conv_state)
    return conv_state, False


def get_retriever() -> HybridRetriever:
    """Get or create the singleton HybridRetriever."""
    global retriever
    if retriever is None:
        retriever = HybridRetriever(
            _gold_path, _silver_path, mode=_mode,
            cosmos_adapter=_cosmos_adapter,
        )
    return retriever


def generate_examples() -> list[list[str]]:
    """
    Generate example questions from Gold layer data:
    community summaries, top entities, and thread subjects.
    Falls back to generic questions if data is unavailable.
    """
    examples = []
    gold = Path(_gold_path)
    silver = Path(_silver_path)
    cosmos = _cosmos_adapter

    try:
        # 1. Community-based questions — pick top 2 communities by entity count
        if cosmos and cosmos.is_configured:
            communities = cosmos.get_communities_by_level(0)
        else:
            communities = []
            comm_dirs = sorted(gold.glob("communities/level_*"))
            for d in comm_dirs:
                for f in d.glob("*.json"):
                    with open(f) as fh:
                        communities.append(json.load(fh))
        communities.sort(key=lambda c: c.get("entity_count", 0), reverse=True)

        for comm in communities[:2]:
            summary = comm.get("summary", "")
            key_ents = [e["name"] for e in comm.get("key_entities", [])[:3]]
            if summary and key_ents:
                ent_str = ", ".join(key_ents)
                first_sent = summary.split(".")[0].strip()
                if len(first_sent) > 20:
                    examples.append([f"What is the relationship between {ent_str}?"])

        # 2. Entity-based questions — top ORGs and PRODUCTs by mention count
        nodes = {}
        if cosmos and cosmos.is_configured and cosmos.gremlin_endpoint:
            try:
                for ntype in cosmos.list_node_types():
                    results = cosmos._gremlin_query(
                        f"g.V().has('node_type', '{ntype}').valueMap(true).limit(500)"
                    )
                    for raw in results:
                        parsed = cosmos._parse_gremlin_node(raw)
                        nid = parsed.get("node_id", "")
                        nodes[nid] = parsed
            except Exception:
                pass
        else:
            nodes_file = gold / "knowledge_graph" / "nodes.json"
            if nodes_file.exists():
                with open(nodes_file) as fh:
                    nodes = json.load(fh)

        if nodes:
            orgs = sorted(
                [n for n in nodes.values() if n.get("node_type") == "ORG"],
                key=lambda n: n.get("mention_count", 0), reverse=True,
            )
            if orgs:
                examples.append([f"What do the emails say about {orgs[0]['name']}?"])

            _noise = lambda name: (
                any(c.isdigit() for c in name[:3])
                or "." in name or ":" in name or "@" in name
                or len(name) < 4 or len(name) > 50
            )
            docs = sorted(
                [n for n in nodes.values()
                 if n.get("node_type") in ("PRODUCT", "WORK_OF_ART", "LAW")
                 and not _noise(n.get("name", ""))],
                key=lambda n: n.get("mention_count", 0), reverse=True,
            )
            if docs:
                top_docs = ", ".join(d["name"] for d in docs[:3])
                examples.append([f"What are {top_docs} and how are they used?"])

            persons = sorted(
                [n for n in nodes.values() if n.get("node_type") == "PERSON"],
                key=lambda n: n.get("mention_count", 0), reverse=True,
            )
            if persons:
                examples.append([f"What tasks or projects is {persons[0]['name']} involved in?"])

        # 3. Thread-subject based question — pick a specific thread topic
        if cosmos and cosmos.is_configured:
            try:
                container = cosmos._get_container("thread_summaries")
                query = "SELECT TOP 5 * FROM c ORDER BY c.email_count DESC"
                summaries = list(container.query_items(query=query, enable_cross_partition_query=True))
            except Exception:
                summaries = []
        else:
            summaries = []
            summary_dir = silver / "not_personal" / "thread_summaries"
            if summary_dir.exists():
                for f in summary_dir.glob("*.json"):
                    with open(f) as fh:
                        summaries.append(json.load(fh))
                summaries.sort(key=lambda s: s.get("email_count", 0), reverse=True)

        if summaries:
            subj = summaries[0].get("subject", "").strip()
            if subj:
                examples.append([f"Summarize the email thread about '{subj}'"])

    except Exception as e:
        logger.warning(f"Failed to generate dynamic examples: {e}")

    seen = set()
    unique = []
    for ex in examples:
        if ex[0] not in seen:
            seen.add(ex[0])
            unique.append(ex)
    examples = unique[:6]

    return examples if examples else FALLBACK_EXAMPLES


# ---------------------------------------------------------------------------
# Query rewriter for multi-turn
# ---------------------------------------------------------------------------

def rewrite_query(
    question: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Rewrite a follow-up question into a standalone query using conversation history.

    If the question is already self-contained (first turn, or no pronouns/references),
    returns it unchanged. Otherwise uses LLM to resolve references.
    """
    if not history:
        return question

    # Quick heuristic: skip rewriter if query seems self-contained
    # (has named entities and no obvious references)
    follow_up_signals = [
        "it", "they", "them", "this", "that", "those", "these",
        "the same", "more about", "what else", "who else",
        "how about", "and what", "tell me more", "expand on",
        "which one", "the first", "the second", "the last",
    ]
    question_lower = question.lower().strip()
    is_follow_up = any(signal in question_lower for signal in follow_up_signals)

    if not is_follow_up and len(question.split()) > 4:
        return question

    # Build history string from full conversation state (already compacted if needed)
    history_text = build_history_text(history)

    # Use LLM to rewrite
    r = get_retriever()
    if not r.llm_client:
        return question

    system_prompt = get_prompt("retrieval", "query_rewriter", "system_prompt", "")
    user_prompt = format_prompt(
        get_prompt("retrieval", "query_rewriter", "user_prompt", ""),
        history=history_text,
        question=question,
    )

    try:
        response = r.llm_client.chat.completions.create(
            model=r.config.answer_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=get_prompt("retrieval", "query_rewriter", "temperature", 0.0),
            max_tokens=get_prompt("retrieval", "query_rewriter", "max_tokens", 200),
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")

    return question


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_answer(result: RetrievalResult) -> str:
    if not result.answer:
        return "*No answer generated — the LLM may be unavailable or no chunks were retrieved.*"
    return result.answer


def format_sources(result: RetrievalResult) -> str:
    if not result.chunks:
        return "*No source chunks retrieved.*"

    lines = ["### Sources\n"]
    for i, chunk in enumerate(result.chunks[:10], 1):
        chunk_id = chunk.get("chunk_id", "?")
        thread = chunk.get("thread_subject", "")
        source_type = chunk.get("source_type", "email")
        sim = chunk.get("similarity_score")
        score_str = f" | score: {sim:.3f}" if sim is not None else ""

        lines.append(f"**[{i}]** {source_type} | {thread}{score_str}")
        lines.append(f"  `{chunk_id}`")
        lines.append("")
    return "\n".join(lines)


def format_metadata(result: RetrievalResult, rewritten_query: str = "", original_query: str = "") -> str:
    grounded_str = "Yes" if result.is_grounded else "No"
    steps = result.metadata.get("steps", 0)
    total_tokens = result.metadata.get("total_tokens", 0)

    lines = [
        "### Metadata\n",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Strategy | **{result.strategy}** |",
        f"| Confidence | {result.confidence:.2%} |",
        f"| Chunks | {len(result.chunks)} |",
        f"| Time | {result.execution_time:.2f}s |",
        f"| Steps | {steps} |",
        f"| Tokens | {total_tokens} |",
    ]
    # Rewritten query shown separately below the input, not in metadata
    # Rewritten query shown separately below the input, not in metadata

    return "\n".join(lines)


def format_react_trace(result: RetrievalResult) -> str:
    """Format ReAct reasoning steps as a markdown table."""
    steps = result.metadata.get("reasoning_trace", [])
    if not steps:
        return "*No reasoning trace available.*"

    lines = [
        "### ReAct Reasoning Trace\n",
        "| Step | Thought | Action | Observation |",
        "|------|---------|--------|-------------|",
    ]
    for s in steps:
        thought = (s.get("thought") or "")[:120].replace("|", "\\|").replace("\n", " ")
        action = s.get("action") or "—"
        obs = (s.get("observation") or "")[:100].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {s.get('step_number', '?')} | {thought} | {action} | {obs} |")

    total_tokens = result.metadata.get("total_tokens", 0)
    if total_tokens:
        lines.append(f"\n*Total tokens: {total_tokens}*")

    return "\n".join(lines)


def format_sources_compact(result: RetrievalResult) -> str:
    """Compact source list for chat detail panel."""
    if not result.chunks:
        return ""
    lines = []
    for i, chunk in enumerate(result.chunks[:5], 1):
        thread = chunk.get("thread_subject", "")
        source_type = chunk.get("source_type", "email")
        lines.append(f"{i}. [{source_type}] {thread}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat handler (multi-turn)
# ---------------------------------------------------------------------------

def chat_respond(
    message: str,
    chat_history: List[Dict[str, str]],
    conv_state: List[Dict[str, str]],
    strategy_label: str,
    top_k: int,
):
    """
    Handle a chat message with multi-turn context.

    Args:
        message: User's new message
        chat_history: Gradio chatbot display history [{"role": ..., "content": ...}, ...]
        conv_state: Internal conversation state [{"user": ..., "answer": ...}, ...]
        strategy_label: Selected retrieval strategy
        top_k: Number of chunks to retrieve

    Returns:
        (updated_chat_history, updated_conv_state, sources_md, metadata_md, rewrite_display)
    """
    if not message.strip():
        return chat_history, conv_state, "", "", "", ""

    strategy = STRATEGY_MAP[strategy_label]
    r = get_retriever()
    r.config.top_k_per_strategy = int(top_k)
    r.config.final_top_k = int(top_k)

    # Step 1: Compact history if approaching token budget
    conv_state, was_compacted = maybe_compact(conv_state)

    # Step 2: Rewrite follow-up query using conversation history
    rewritten = rewrite_query(message, conv_state)
    rewrite_display = ""
    if rewritten != message:
        rewrite_display = f"*Rewritten:* {rewritten}"

    # Step 3: Retrieve using the (possibly rewritten) query
    result = r.retrieve(rewritten, strategy, conversation_history=build_history_text(conv_state))

    # Step 4: Format response
    answer = format_answer(result)
    sources = format_sources(result)
    metadata = format_metadata(result, rewritten_query=rewritten, original_query=message)

    if strategy == RetrievalStrategy.REACT:
        trace = format_react_trace(result)
        if trace and "No reasoning trace" not in trace:
            metadata += "\n\n" + trace

    # Step 5: Update conversation state
    conv_state.append({
        "user": message,
        "rewritten": rewritten,
        "answer": result.answer or "",
        "strategy": strategy_label,
        "sources_count": len(result.chunks),
    })

    # Step 6: Update chat display (messages format for Gradio 6.x)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    # Step 7: Signal compaction to user
    compaction_notice = ""
    if was_compacted:
        compaction_notice = "Earlier conversation was summarized to maintain quality."

    return chat_history, conv_state, sources, metadata, rewrite_display, compaction_notice


def clear_chat():
    """Reset chat history and conversation state."""
    return [], [], "", "", "", ""


# ---------------------------------------------------------------------------
# Single query handler (stateless, kept for detailed inspection)
# ---------------------------------------------------------------------------

def ask_single(question: str, strategy_label: str, top_k: int):
    """Handle a single-strategy query."""
    if not question.strip():
        return "Please enter a question.", "", "", ""

    strategy = STRATEGY_MAP[strategy_label]
    r = get_retriever()
    r.config.top_k_per_strategy = int(top_k)
    r.config.final_top_k = int(top_k)

    result = r.retrieve(question, strategy)

    react_trace = ""
    if strategy == RetrievalStrategy.REACT:
        react_trace = format_react_trace(result)

    return format_answer(result), format_sources(result), format_metadata(result), react_trace


# ---------------------------------------------------------------------------
# Compare handler
# ---------------------------------------------------------------------------

def compare_strategies(question: str, selected_labels: list[str]):
    """Run selected strategies and return comparison table + per-strategy details."""
    if not question.strip():
        return "", ""
    if not selected_labels:
        return "*Select at least one strategy.*", ""

    r = get_retriever()
    results: dict[str, RetrievalResult] = {}

    for label in selected_labels:
        strategy = STRATEGY_MAP[label]
        results[label] = r.retrieve(question, strategy)

    table_lines = [
        "### Comparison Summary\n",
        "| Strategy | Chunks | Confidence | Grounded | Time (s) |",
        "|----------|--------|------------|----------|----------|",
    ]
    for label, res in results.items():
        grounded = "Yes" if res.is_grounded else "No"
        table_lines.append(
            f"| **{label}** | {len(res.chunks)} | {res.confidence:.2%} | {grounded} | {res.execution_time:.2f} |"
        )

    detail_lines = ["\n---\n"]
    for label, res in results.items():
        detail_lines.append(f"## {label}\n")
        detail_lines.append(format_answer(res))
        detail_lines.append("")
        detail_lines.append(format_sources(res))
        if label == "ReAct":
            detail_lines.append(format_react_trace(res))
        detail_lines.append("\n---\n")

    return "\n".join(table_lines), "\n".join(detail_lines)


def export_comparison_json(question: str, selected_labels: list[str]) -> str | None:
    """Run strategies and return downloadable JSON."""
    if not question.strip() or not selected_labels:
        return None

    r = get_retriever()
    export = {"query": question, "results": {}}

    for label in selected_labels:
        strategy = STRATEGY_MAP[label]
        result = r.retrieve(question, strategy)
        export["results"][label] = result.to_dict()

    out_path = Path("/tmp/strategy_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=str)
    return str(out_path)


# ---------------------------------------------------------------------------
# Graph stats
# ---------------------------------------------------------------------------

def get_graph_stats() -> str:
    try:
        r = get_retriever()
        toolkit = r.toolkit
        graph = toolkit._load_graph()

        comm_dir = toolkit.gold_path / "communities"
        comm_count = 0
        if comm_dir.exists():
            comm_count = sum(1 for _ in comm_dir.rglob("*.json"))

        path_file = toolkit.gold_path / "paths" / "path_index.json"
        path_count = 0
        if path_file.exists():
            with open(path_file, "r") as f:
                path_count = len(json.load(f))

        emb_dir = toolkit.gold_path / "embeddings"
        chunk_emb = 0
        entity_emb = 0
        if emb_dir.exists():
            chunk_emb_file = emb_dir / "chunks_embeddings.npy"
            entity_emb_file = emb_dir / "entities_embeddings.npy"
            if chunk_emb_file.exists():
                import numpy as np
                chunk_emb = len(np.load(str(chunk_emb_file)))
            if entity_emb_file.exists():
                import numpy as np
                entity_emb = len(np.load(str(entity_emb_file)))

        return f"""### Knowledge Graph Statistics
| Metric | Value |
|--------|-------|
| Nodes | {graph.number_of_nodes():,} |
| Edges | {graph.number_of_edges():,} |
| Communities | {comm_count} |
| Paths | {path_count:,} |
| Chunk Embeddings | {chunk_emb:,} |
| Entity Embeddings | {entity_emb:,} |
"""
    except Exception as e:
        return f"*Could not load graph stats: {e}*"


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

APP_CSS = """
.container { max-width: 1400px; margin: auto; }
.answer-box { min-height: 150px; }
button[aria-label="share"], button[aria-label="delete"] { display: none !important; }
"""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Email Knowledge Graph — Retrieval") as app:
        gr.Markdown(
            """
            # Email Knowledge Graph — Retrieval
            Multi-turn chat and strategy comparison over the enterprise email knowledge graph.
            """
        )

        # ========================= Tab 1: Query (multi-turn chat) ===============
        with gr.Tab("Query"):
            # Hidden state for conversation history
            conv_state = gr.State([])

            # Strategy + controls at the top
            with gr.Row():
                chat_strategy = gr.Radio(
                    choices=STRATEGY_LABELS,
                    value="Hybrid",
                    label="Strategy",
                )
                chat_top_k = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Top-k",
                )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=480,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask a question (follow-ups supported)…",
                            show_label=False,
                            scale=5,
                            lines=1,
                        )
                        chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        gr.Column(scale=5)  # spacer
                        chat_clear_btn = gr.Button("Clear", variant="secondary", size="sm", scale=1)

                    rewrite_display = gr.Markdown()
                    compaction_notice = gr.Markdown()

                with gr.Column(scale=2):
                    chat_sources = gr.Markdown(label="Sources")
                    chat_metadata = gr.Markdown(label="Metadata")

            gr.Examples(
                examples=generate_examples(),
                inputs=[chat_input],
                label="Example Questions",
            )

            # Chat events
            chat_outputs = [chatbot, conv_state, chat_sources, chat_metadata, rewrite_display, compaction_notice]
            chat_send_btn.click(
                fn=chat_respond,
                inputs=[chat_input, chatbot, conv_state, chat_strategy, chat_top_k],
                outputs=chat_outputs,
            ).then(
                fn=lambda: "",
                outputs=[chat_input],
            )
            chat_input.submit(
                fn=chat_respond,
                inputs=[chat_input, chatbot, conv_state, chat_strategy, chat_top_k],
                outputs=chat_outputs,
            ).then(
                fn=lambda: "",
                outputs=[chat_input],
            )
            chat_clear_btn.click(
                fn=clear_chat,
                outputs=chat_outputs,
            )

        # ========================= Tab 3: Compare Strategies ====================
        with gr.Tab("Compare Strategies"):
            compare_question = gr.Textbox(
                label="Question",
                placeholder="Ask a question to compare across strategies…",
                lines=2,
            )
            strategy_checks = gr.CheckboxGroup(
                choices=STRATEGY_LABELS,
                value=STRATEGY_LABELS,
                label="Strategies to compare",
            )
            with gr.Row():
                compare_btn = gr.Button("Compare", variant="primary")
                export_btn = gr.Button("Export JSON", variant="secondary")

            summary_output = gr.Markdown(label="Summary")
            details_output = gr.Markdown(label="Per-Strategy Details")
            export_file = gr.File(label="Download JSON", visible=False)

            compare_btn.click(
                fn=compare_strategies,
                inputs=[compare_question, strategy_checks],
                outputs=[summary_output, details_output],
            )
            compare_question.submit(
                fn=compare_strategies,
                inputs=[compare_question, strategy_checks],
                outputs=[summary_output, details_output],
            )
            export_btn.click(
                fn=export_comparison_json,
                inputs=[compare_question, strategy_checks],
                outputs=[export_file],
            )

        gr.Markdown(
            """
            ---
            **Pipeline**: Bronze → Silver → Gold | **Strategies**: Vector · PathRAG · GraphRAG · Hybrid · ReAct
            """
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Gradio app for retrieval strategy comparison",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["local", "llm", "hybrid"],
        help="Processing mode — auto-derives silver/gold paths",
    )
    parser.add_argument("--gold", type=str, help="Path to Gold layer")
    parser.add_argument("--silver", type=str, help="Path to Silver layer")
    parser.add_argument("--port", type=int, default=7861, help="Server port (default: 7861)")
    return parser.parse_args()


def main():
    global _gold_path, _silver_path, _mode, _cosmos_adapter

    args = parse_args()
    data_root = Path("./data")

    if args.gold:
        _gold_path = args.gold
    elif args.mode:
        _gold_path = str(data_root / f"gold_{args.mode}")
    else:
        _gold_path = str(data_root / "gold_local")

    if args.silver:
        _silver_path = args.silver
    elif args.mode:
        _silver_path = str(data_root / f"silver_{args.mode}")
    else:
        _silver_path = str(data_root / "silver_local")

    _mode = args.mode or "local"

    # Initialize Cosmos DB adapter if configured
    if os.getenv("COSMOS_GREMLIN_ENDPOINT") or os.getenv("COSMOS_NOSQL_ENDPOINT"):
        from src.storage.cosmos_adapter import CosmosAdapter
        _cosmos_adapter = CosmosAdapter.from_env()
        logger.info(f"Cosmos DB adapter initialized (configured={_cosmos_adapter.is_configured})")

    if not _cosmos_adapter and not Path(_gold_path).exists():
        logger.error(f"Gold path does not exist: {_gold_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Email Knowledge Graph — Retrieval")
    print("=" * 60)
    print(f"  Mode:   {_mode}")
    print(f"  Gold:   {_gold_path}")
    print(f"  Silver: {_silver_path}")
    print(f"  Cosmos: {'configured' if _cosmos_adapter else 'not configured (local mode)'}")
    print(f"  Port:   {args.port}")

    print("\nLoading retriever…")
    get_retriever()
    print("Retriever loaded!")

    app = create_app()

    print(f"\nStarting Gradio server…")
    print(f"Open http://localhost:{args.port} in your browser")
    print("=" * 60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
