#!/usr/bin/env python3
"""
Retrieval Strategy Comparison — Gradio Web Interface

Interactive comparison of 5 retrieval strategies:
  Vector | PathRAG | GraphRAG | Hybrid | ReAct

Usage:
    python -m src.app --mode local
    python -m src.app --gold ./data/gold_local --silver ./data/silver_local
    python -m src.app --port 7861
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import gradio as gr

from src.retrieval import HybridRetriever, RetrievalStrategy, RetrievalResult

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

STRATEGY_LABELS = ["Vector", "PathRAG", "GraphRAG", "Hybrid", "ReAct"]
STRATEGY_MAP = {
    "Vector": RetrievalStrategy.VECTOR,
    "PathRAG": RetrievalStrategy.PATHRAG,
    "GraphRAG": RetrievalStrategy.GRAPHRAG,
    "Hybrid": RetrievalStrategy.HYBRID,
    "ReAct": RetrievalStrategy.REACT,
}

EXAMPLES = [
    ["What projects were discussed in the emails and who was involved?"],
    ["What mortgage-related documents were processed?"],
    ["Who are the main contacts and what roles do they play?"],
    ["What compliance or regulatory topics appear in the archive?"],
    ["What technical issues were reported and how were they resolved?"],
]


def get_retriever() -> HybridRetriever:
    """Get or create the singleton HybridRetriever."""
    global retriever
    if retriever is None:
        retriever = HybridRetriever(_gold_path, _silver_path)
    return retriever


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
        sim = chunk.get("similarity_score")
        score_str = f" | score: {sim:.4f}" if sim is not None else ""
        lines.append(f"**[{i}]** `{chunk_id}`{score_str}")
        if thread:
            lines.append(f"  Thread: {thread}")
        lines.append("")
    return "\n".join(lines)


def format_metadata(result: RetrievalResult) -> str:
    grounded_str = "Yes" if result.is_grounded else "No"
    lines = [
        "### Metadata\n",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Strategy | **{result.strategy}** |",
        f"| Grounded | {grounded_str} |",
        f"| Confidence | {result.confidence:.2%} |",
        f"| Chunks | {len(result.chunks)} |",
        f"| Time | {result.execution_time:.2f}s |",
    ]
    if result.missing_info:
        lines.append(f"| Missing Info | {result.missing_info} |")

    # Extra metadata keys
    skip = {"reasoning_trace", "tool_message"}
    for key, val in result.metadata.items():
        if key in skip:
            continue
        # Compact display for complex values
        display = val if isinstance(val, (int, float, str, bool)) else json.dumps(val, default=str)[:120]
        lines.append(f"| {key} | {display} |")

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


# ---------------------------------------------------------------------------
# Query handlers
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

    # Summary table
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

    # Per-strategy details
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

    # Write temp file for download
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

        # Count community files
        comm_dir = toolkit.gold_path / "communities"
        comm_count = 0
        if comm_dir.exists():
            comm_count = sum(1 for _ in comm_dir.rglob("*.json"))

        # Count paths
        path_file = toolkit.gold_path / "paths" / "path_index.json"
        path_count = 0
        if path_file.exists():
            with open(path_file, "r") as f:
                path_count = len(json.load(f))

        # Count embeddings
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
"""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Retrieval Strategy Comparison") as app:
        gr.Markdown(
            """
            # Retrieval Strategy Comparison
            Compare **Vector**, **PathRAG**, **GraphRAG**, **Hybrid**, and **ReAct** retrieval
            over the anonymised email knowledge graph.
            """
        )

        # ========================= Tab 1: Single Query =========================
        with gr.Tab("Single Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    strategy_radio = gr.Radio(
                        choices=STRATEGY_LABELS,
                        value="Hybrid",
                        label="Strategy",
                    )
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the email archive…",
                        lines=2,
                    )
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=20, value=10, step=1,
                            label="Top-k chunks",
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                    answer_output = gr.Markdown(label="Answer", elem_classes=["answer-box"])

                with gr.Column(scale=2):
                    sources_output = gr.Markdown(label="Sources")
                    meta_output = gr.Markdown(label="Metadata")

            with gr.Accordion("ReAct Reasoning Trace", open=False):
                react_output = gr.Markdown()

            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input],
                label="Example Questions",
            )

            # Events
            ask_btn.click(
                fn=ask_single,
                inputs=[question_input, strategy_radio, top_k_slider],
                outputs=[answer_output, sources_output, meta_output, react_output],
            )
            question_input.submit(
                fn=ask_single,
                inputs=[question_input, strategy_radio, top_k_slider],
                outputs=[answer_output, sources_output, meta_output, react_output],
            )

        # ========================= Tab 2: Compare Strategies ====================
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

        # ========================= Footer: Graph Stats =========================
        with gr.Accordion("Knowledge Graph Statistics", open=False):
            stats_output = gr.Markdown(value=get_graph_stats)
            refresh_btn = gr.Button("Refresh", size="sm")
            refresh_btn.click(fn=get_graph_stats, outputs=stats_output)

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
    global _gold_path, _silver_path

    args = parse_args()
    data_root = Path("./data")

    # Resolve gold path
    if args.gold:
        _gold_path = args.gold
    elif args.mode:
        _gold_path = str(data_root / f"gold_{args.mode}")
    else:
        # Default to local
        _gold_path = str(data_root / "gold_local")

    # Resolve silver path
    if args.silver:
        _silver_path = args.silver
    elif args.mode:
        _silver_path = str(data_root / f"silver_{args.mode}")
    else:
        _silver_path = str(data_root / "silver_local")

    if not Path(_gold_path).exists():
        logger.error(f"Gold path does not exist: {_gold_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Retrieval Strategy Comparison")
    print("=" * 60)
    print(f"  Gold:   {_gold_path}")
    print(f"  Silver: {_silver_path}")
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
