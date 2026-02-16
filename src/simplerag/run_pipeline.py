#!/usr/bin/env python3
"""
SimpleRAG Pipeline Runner

End-to-End AI Data Pipeline: Ingestion → Bronze → Silver → Gold (RAG)

Usage:
    # Run full pipeline
    python -m src.simplerag.run_pipeline --all

    # Run individual layers
    python -m src.simplerag.run_pipeline --bronze  # Ingest new files
    python -m src.simplerag.run_pipeline --silver  # Process Bronze → Silver
    python -m src.simplerag.run_pipeline --gold    # Index Silver → Gold

    # Query
    python -m src.simplerag.run_pipeline --query "What is X?"

    # Interactive mode
    python -m src.simplerag.run_pipeline --interactive
"""

import asyncio
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.simplerag.utils.config import SimpleRAGConfig
from src.simplerag.utils.lineage import LineageTracker
from src.simplerag.bronze.ingestion import BronzeIngestion
from src.simplerag.silver.processor import SilverProcessor
from src.simplerag.gold.rag_retriever import GoldRAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRAGPipeline:
    """
    End-to-End SimpleRAG Pipeline.

    Orchestrates: Bronze → Silver → Gold (RAG)

    All outputs preserve lineage to Bronze.
    All AI operations record model info.
    """

    def __init__(self, base_dir: str = "./data/simplerag"):
        """Initialize the pipeline."""
        # Configuration
        self.config = SimpleRAGConfig()
        self.config.directories.base_dir = Path(base_dir)
        self.config.directories.__post_init__()  # Reinitialize paths
        self.config.initialize()

        # Lineage tracker
        self.lineage = LineageTracker(
            self.config.directories.gold_dir / "metadata" / "lineage"
        )

        # Layer components
        self.bronze = BronzeIngestion(self.config)
        self.silver = SilverProcessor(self.config, self.lineage)
        self.gold = GoldRAGRetriever(self.config, self.lineage)

        logger.info(f"SimpleRAG pipeline initialized: {base_dir}")

    async def run_bronze(self) -> dict:
        """
        Run Bronze layer: Ingest new files.

        NO AI - just raw extraction.
        """
        logger.info("=" * 60)
        logger.info("BRONZE LAYER: Raw Ingestion")
        logger.info("=" * 60)

        start = datetime.now()
        records = self.bronze.process_source_folder()
        duration = (datetime.now() - start).total_seconds()

        result = {
            "layer": "bronze",
            "records_ingested": len(records),
            "duration_seconds": duration,
            "record_ids": [r.record_id for r in records]
        }

        logger.info(f"Bronze complete: {len(records)} records in {duration:.2f}s")
        return result

    async def run_silver(self) -> dict:
        """
        Run Silver layer: Process Bronze → Silver.

        Includes OCR, anonymization, summarization.
        """
        logger.info("=" * 60)
        logger.info("SILVER LAYER: OCR, Anonymization, Summarization")
        logger.info("=" * 60)

        start = datetime.now()
        records = await self.silver.process_all_bronze()
        duration = (datetime.now() - start).total_seconds()

        result = {
            "layer": "silver",
            "records_processed": len(records),
            "duration_seconds": duration,
            "record_ids": [r.record_id for r in records],
            "total_chunks": sum(len(r.chunks) for r in records)
        }

        logger.info(f"Silver complete: {len(records)} records, {result['total_chunks']} chunks in {duration:.2f}s")
        return result

    async def run_gold(self) -> dict:
        """
        Run Gold layer: Index Silver → Gold.

        Creates embeddings and search index.
        Skips already indexed chunks.
        """
        logger.info("=" * 60)
        logger.info("GOLD LAYER: Indexing for RAG")
        logger.info("=" * 60)

        start = datetime.now()

        # Index all Silver records (skips existing)
        await self.gold.index_all_silver(skip_existing=True)

        duration = (datetime.now() - start).total_seconds()
        stats = self.gold.get_stats()

        result = {
            "layer": "gold",
            "duration_seconds": duration,
            **stats
        }

        logger.info(f"Gold complete: {stats['total_chunks']} chunks, {stats['unique_emails']} emails indexed in {duration:.2f}s")
        return result

    async def run_all(self) -> dict:
        """Run full pipeline: Bronze → Silver → Gold."""
        logger.info("=" * 60)
        logger.info("RUNNING FULL PIPELINE")
        logger.info("=" * 60)

        results = {
            "pipeline": "simplerag",
            "start_time": datetime.now().isoformat(),
            "layers": {}
        }

        # Bronze
        results["layers"]["bronze"] = await self.run_bronze()

        # Silver
        results["layers"]["silver"] = await self.run_silver()

        # Gold
        results["layers"]["gold"] = await self.run_gold()

        results["end_time"] = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results

    async def query(self, question: str) -> dict:
        """
        Query the RAG system.

        Returns grounded answer with lineage.
        """
        response = await self.gold.query(question)
        return response.to_dict()

    async def interactive(self):
        """Interactive query mode."""
        print("\n" + "=" * 60)
        print("SimpleRAG Interactive Mode")
        print("=" * 60)
        print("Type 'quit' to exit, 'stats' for index stats")
        print("Type 'lineage <record_id>' to trace lineage")
        print()

        while True:
            try:
                user_input = input("Query: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if user_input.lower() == 'stats':
                    stats = self.gold.get_index_stats()
                    print(f"\nIndex Stats: {json.dumps(stats, indent=2)}\n")
                    continue

                if user_input.lower().startswith('lineage '):
                    record_id = user_input.split(' ', 1)[1]
                    trace = self.lineage.trace_to_bronze(record_id)
                    if trace:
                        print(f"\nLineage: {json.dumps(trace, indent=2)}\n")
                    else:
                        print(f"\nNo lineage found for: {record_id}\n")
                    continue

                # Query
                response = await self.query(user_input)

                print(f"\nAnswer: {response['answer']}")
                print(f"\nGrounded: {response['is_grounded']}")
                print(f"Confidence: {response['confidence']:.2f}")
                print(f"Sources: {len(response['sources'])}")

                if response.get('missing_info'):
                    print(f"Missing Info: {response['missing_info']}")

                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")

    async def close(self):
        """Close async resources."""
        await self.silver.close()


async def main():
    parser = argparse.ArgumentParser(description="SimpleRAG Pipeline")
    parser.add_argument("--base-dir", default="./data/simplerag", help="Base directory")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--bronze", action="store_true", help="Run Bronze layer only")
    parser.add_argument("--silver", action="store_true", help="Run Silver layer only")
    parser.add_argument("--gold", action="store_true", help="Run Gold layer only")
    parser.add_argument("--query", "-q", type=str, help="Query to run")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    pipeline = SimpleRAGPipeline(args.base_dir)

    try:
        if args.all:
            results = await pipeline.run_all()
            print(f"\nResults: {json.dumps(results, indent=2)}")

        elif args.bronze:
            results = await pipeline.run_bronze()
            print(f"\nResults: {json.dumps(results, indent=2)}")

        elif args.silver:
            results = await pipeline.run_silver()
            print(f"\nResults: {json.dumps(results, indent=2)}")

        elif args.gold:
            results = await pipeline.run_gold()
            print(f"\nResults: {json.dumps(results, indent=2)}")

        elif args.query:
            response = await pipeline.query(args.query)
            print(f"\nQuery: {args.query}")
            print(f"Answer: {response['answer']}")
            print(f"\nGrounded: {response['is_grounded']}")
            print(f"Sources: {len(response['sources'])}")

        elif args.interactive:
            await pipeline.interactive()

        else:
            parser.print_help()

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
