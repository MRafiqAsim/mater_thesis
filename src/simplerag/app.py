#!/usr/bin/env python3
"""
SimpleRAG Gradio Web Interface

Usage:
    python -m src.simplerag.app

Opens a web interface at http://localhost:7860
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import gradio as gr
from src.simplerag.run_pipeline import SimpleRAGPipeline

# Initialize pipeline (singleton)
pipeline = None


def get_pipeline():
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = SimpleRAGPipeline()
    return pipeline


def get_stats():
    """Get index statistics."""
    p = get_pipeline()
    stats = p.gold.get_stats()
    return f"""### Index Statistics
| Metric | Value |
|--------|-------|
| Total Chunks | {stats['total_chunks']} |
| Total Embeddings | {stats['total_embeddings']} |
| Embedding Dimensions | {stats['embedding_dimensions']} |
| Unique Emails | {stats['unique_emails']} |
| Unique Threads | {stats['unique_threads']} |
| Unique Domains | {stats['unique_domains']} |
"""


async def ask_question_async(question: str, top_k: int = 10):
    """Ask a question and get response."""
    if not question.strip():
        return "Please enter a question.", "", ""

    p = get_pipeline()
    p.config.top_k = top_k

    result = await p.query(question)

    # Format answer
    answer = result['answer']

    # Format sources
    sources_md = "### Sources\n\n"
    if result['sources']:
        for i, src in enumerate(result['sources'], 1):
            subject = src.get('subject', 'N/A')
            timestamp = src.get('timestamp', '')[:10] if src.get('timestamp') else 'N/A'
            score = src.get('score', 0)
            email_id = src.get('email_id', 'N/A')

            sources_md += f"**[{i}]** {subject}\n"
            sources_md += f"- Date: {timestamp} | Score: {score:.4f}\n"
            sources_md += f"- Email ID: `{email_id}`\n\n"
    else:
        sources_md += "_No sources found_"

    # Format metadata
    meta_md = f"""### Response Metadata
- **Grounded**: {'Yes' if result['is_grounded'] else 'No'}
- **Confidence**: {result['confidence']:.2%}
- **Sources Count**: {len(result['sources'])}
- **Model**: {result['model_info'].get('model', 'N/A')}
"""

    return answer, sources_md, meta_md


def ask_question(question: str, top_k: int = 10):
    """Sync wrapper for async question."""
    return asyncio.run(ask_question_async(question, top_k))


# Example questions
EXAMPLES = [
    ["What JIRA issues were created?"],
    ["What VPN access was requested?"],
    ["Tell me about wood work in the office"],
    ["What defects were fixed?"],
    ["Who requested credentials?"],
]


# Create Gradio interface
def create_app():
    """Create Gradio app."""

    with gr.Blocks(
        title="SimpleRAG - Email Knowledge Base",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .answer-box { min-height: 150px; }
        """
    ) as app:

        gr.Markdown("""
        # SimpleRAG - Email Knowledge Base

        Ask questions about the indexed email archive. Answers are grounded in source documents with full lineage tracking.

        **Features**: Semantic search | Source citations | Lineage to Bronze layer
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the email archive...",
                    lines=2
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Sources (top_k)"
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)

                # Answer output
                answer_output = gr.Markdown(
                    label="Answer",
                    elem_classes=["answer-box"]
                )

            with gr.Column(scale=2):
                # Sources
                sources_output = gr.Markdown(label="Sources")

                # Metadata
                meta_output = gr.Markdown(label="Metadata")

        # Examples
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input],
            label="Example Questions"
        )

        # Stats accordion
        with gr.Accordion("Index Statistics", open=False):
            stats_output = gr.Markdown(value=get_stats)
            refresh_btn = gr.Button("Refresh Stats", size="sm")
            refresh_btn.click(fn=get_stats, outputs=stats_output)

        # Event handlers
        submit_btn.click(
            fn=ask_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, meta_output]
        )

        question_input.submit(
            fn=ask_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, meta_output]
        )

        gr.Markdown("""
        ---
        **SimpleRAG Pipeline** | Bronze → Silver → Gold | User Interface
        """)

    return app


def main():
    """Launch the app."""
    print("\n" + "="*60)
    print("SimpleRAG Web Interface")
    print("="*60)

    # Initialize pipeline
    print("Loading pipeline...")
    get_pipeline()
    print("Pipeline loaded!")

    # Create and launch app
    app = create_app()

    print("\nStarting Gradio server...")
    print("Open http://localhost:7860 in your browser")
    print("="*60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
