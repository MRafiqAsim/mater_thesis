#!/usr/bin/env python3
"""
SimpleRAG Query Interface

Usage:
    python -m src.simplerag.ask "Your question here"
    python -m src.simplerag.ask --interactive
    python -m src.simplerag.ask --stats
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.simplerag.run_pipeline import SimpleRAGPipeline


async def single_query(question: str, verbose: bool = False):
    """Ask a single question."""
    pipeline = SimpleRAGPipeline()
    result = await pipeline.query(question)

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    print(result['answer'])
    print(f"\n{'─'*60}")
    print(f"Grounded: {result['is_grounded']} | Confidence: {result['confidence']:.2%} | Sources: {len(result['sources'])}")

    if verbose and result['sources']:
        print(f"\n{'─'*60}")
        print("Sources:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  [{i}] {src['subject']} ({src['timestamp'][:10] if src['timestamp'] else 'N/A'})")
            print(f"      Score: {src['score']:.4f} | Email: {src['email_id']}")


async def interactive_mode():
    """Interactive chat session."""
    pipeline = SimpleRAGPipeline()

    print("\n" + "="*60)
    print("SimpleRAG Interactive Mode")
    print("Commands: 'quit' to exit, 'stats' for index stats")
    print("="*60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if question.lower() == 'stats':
            stats = pipeline.gold.get_stats()
            print(f"\nIndex Stats: {stats}\n")
            continue

        result = await pipeline.query(question)
        print(f"\nAssistant: {result['answer']}")
        print(f"[Sources: {len(result['sources'])} | Confidence: {result['confidence']:.2%}]\n")


async def show_stats():
    """Show index statistics."""
    pipeline = SimpleRAGPipeline()
    stats = pipeline.gold.get_stats()

    print("\n" + "="*60)
    print("SimpleRAG Index Statistics")
    print("="*60)
    print(f"  Total Chunks:     {stats['total_chunks']}")
    print(f"  Total Embeddings: {stats['total_embeddings']}")
    print(f"  Dimensions:       {stats['embedding_dimensions']}")
    print(f"  Unique Emails:    {stats['unique_emails']}")
    print(f"  Unique Threads:   {stats['unique_threads']}")
    print(f"  Unique Domains:   {stats['unique_domains']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SimpleRAG Query Interface")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-s", "--stats", action="store_true", help="Show index stats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output with sources")

    args = parser.parse_args()

    if args.stats:
        asyncio.run(show_stats())
    elif args.interactive:
        asyncio.run(interactive_mode())
    elif args.question:
        asyncio.run(single_query(args.question, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
