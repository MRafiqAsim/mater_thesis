"""
PathRAG Pipeline - Official Implementation

Uses the official PathRAG library for benchmarking.
Converts Silver layer data to PathRAG format.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.pathrag import PathRAG, QueryParam

logger = logging.getLogger(__name__)


@dataclass
class PathRAGPipelineConfig:
    """Configuration for PathRAG pipeline."""
    working_dir: str = "./data/pathrag_index"
    chunk_token_size: int = 1200
    embedding_batch_size: int = 32
    top_k: int = 10


class PathRAGPipeline:
    """
    PathRAG Pipeline using official implementation.

    This pipeline:
    1. Loads Silver layer chunks
    2. Inserts them into PathRAG's index
    3. Provides query interface using PathRAG's hybrid mode
    """

    def __init__(
        self,
        silver_path: str,
        config: Optional[PathRAGPipelineConfig] = None
    ):
        self.silver_path = Path(silver_path)
        self.config = config or PathRAGPipelineConfig()

        # Ensure working directory exists
        os.makedirs(self.config.working_dir, exist_ok=True)

        # Initialize PathRAG
        self.rag = None
        self._initialized = False

        logger.info(f"PathRAGPipeline initialized: {silver_path} -> {self.config.working_dir}")

    def initialize(self):
        """Initialize real PathRAG instance."""
        if self._initialized:
            return

        self.rag = PathRAG(
            working_dir=self.config.working_dir,
            chunk_token_size=self.config.chunk_token_size,
            embedding_batch_num=self.config.embedding_batch_size,
            llm_model_name="gpt-4o",
            enable_llm_cache=True,
        )
        self._initialized = True
        logger.info("PathRAG initialized")

    def load_silver_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from Silver layer."""
        chunks = []

        # Load from thread_chunks
        thread_chunks_path = self.silver_path / "thread_chunks"
        if thread_chunks_path.exists():
            for chunk_file in thread_chunks_path.glob("*.json"):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to load {chunk_file}: {e}")

        # Load from individual_chunks
        individual_chunks_path = self.silver_path / "individual_chunks"
        if individual_chunks_path.exists():
            for chunk_file in individual_chunks_path.glob("*.json"):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to load {chunk_file}: {e}")

        logger.info(f"Loaded {len(chunks)} chunks from Silver layer")
        return chunks

    def convert_chunk_to_pathrag_format(self, chunk: Dict[str, Any]) -> str:
        """Convert a Silver chunk to PathRAG document format."""
        # Use original text if available, otherwise anonymized
        text = chunk.get("text_original", chunk.get("text_anonymized", ""))

        # Add metadata as context
        metadata_parts = []

        if chunk.get("thread_subject"):
            metadata_parts.append(f"Subject: {chunk['thread_subject']}")

        if chunk.get("thread_participants"):
            participants = ", ".join(chunk["thread_participants"][:5])
            metadata_parts.append(f"Participants: {participants}")

        if chunk.get("email_position"):
            metadata_parts.append(f"Position: {chunk['email_position']}")

        # Combine metadata with text
        if metadata_parts:
            metadata_str = " | ".join(metadata_parts)
            return f"[{metadata_str}]\n\n{text}"

        return text

    async def index_documents(self, progress_callback=None) -> Dict[str, Any]:
        """
        Index Silver layer documents into PathRAG.

        Returns:
            Statistics about indexing
        """
        self.initialize()

        chunks = self.load_silver_chunks()

        if not chunks:
            return {"error": "No chunks found in Silver layer"}

        # Convert to PathRAG format
        documents = []
        for chunk in chunks:
            doc = self.convert_chunk_to_pathrag_format(chunk)
            if doc.strip():
                documents.append(doc)

        logger.info(f"Indexing {len(documents)} documents into PathRAG...")

        # Use PathRAG's async insert
        await self.rag.ainsert(documents)

        return {
            "total_chunks": len(chunks),
            "indexed_documents": len(documents),
            "working_dir": self.config.working_dir,
        }

    async def query(
        self,
        question: str,
        mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Query using PathRAG.

        Args:
            question: The query
            mode: Query mode ("hybrid" for PathRAG)

        Returns:
            Query result with answer and metadata
        """
        self.initialize()

        start_time = datetime.now()

        try:
            param = QueryParam(
                mode=mode,
                top_k=self.config.top_k,
            )

            # Use PathRAG's async query
            response = await self.rag.aquery(question, param)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "query": question,
                "answer": response,
                "mode": mode,
                "execution_time": execution_time,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": question,
                "answer": "",
                "error": str(e),
                "success": False,
            }

    def query_sync(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Synchronous query wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.query(question, mode))
        finally:
            loop.close()


def main():
    """CLI for PathRAG pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="PathRAG Pipeline")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument("--working-dir", default="./data/pathrag_index", help="PathRAG working directory")
    parser.add_argument("--index", action="store_true", help="Index documents")
    parser.add_argument("--query", "-q", type=str, help="Query to run")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = PathRAGPipelineConfig(working_dir=args.working_dir)
    pipeline = PathRAGPipeline(args.silver, config)

    if args.index:
        print("Indexing documents into PathRAG...")
        loop = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(pipeline.index_documents())
            print(f"\nIndexing complete: {json.dumps(stats, indent=2)}")
        finally:
            loop.close()

    if args.query:
        print(f"\nQuery: {args.query}")
        result = pipeline.query_sync(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"Time: {result.get('execution_time', 0):.2f}s")

    if args.interactive:
        print("\n" + "="*60)
        print("PathRAG Interactive Mode")
        print("="*60)
        print("Type 'quit' to exit\n")

        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue

                result = pipeline.query_sync(query)
                print(f"\nAnswer: {result['answer']}")
                print(f"Time: {result.get('execution_time', 0):.2f}s\n")

            except KeyboardInterrupt:
                break

        print("Goodbye!")


if __name__ == "__main__":
    main()
