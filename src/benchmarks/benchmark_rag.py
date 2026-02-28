"""
RAG Benchmarking Framework

Benchmarks different RAG approaches:
1. Basic RAG (Vector Search)
2. GraphRAG (Community-based)
3. PathRAG (Official implementation)
4. ReAct + GraphRAG
5. ReAct + PathRAG
6. Hybrid (Combined)
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)


class RAGStrategy(Enum):
    """Available RAG strategies for benchmarking."""
    BASIC_RAG = "basic_rag"  # Vector search only
    GRAPHRAG = "graphrag"  # Community-based
    PATHRAG = "pathrag"  # Official PathRAG
    REACT_GRAPHRAG = "react_graphrag"  # ReAct with GraphRAG tools
    REACT_PATHRAG = "react_pathrag"  # ReAct with PathRAG tools
    HYBRID = "hybrid"  # Combined approach


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected properties."""
    query: str
    category: str  # e.g., "factual", "multi-hop", "temporal", "aggregation"
    difficulty: str  # "easy", "medium", "hard"
    expected_entities: List[str] = field(default_factory=list)
    expected_topics: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None  # Optional ground truth answer


@dataclass
class BenchmarkResult:
    """Result from a single benchmark query."""
    query: str
    strategy: str
    answer: str
    execution_time: float
    chunks_retrieved: int
    confidence: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results for a strategy."""
    strategy: str
    total_queries: int
    successful_queries: int
    avg_execution_time: float
    avg_chunks_retrieved: float
    avg_confidence: float
    queries_by_category: Dict[str, int] = field(default_factory=dict)
    queries_by_difficulty: Dict[str, int] = field(default_factory=dict)


class RAGBenchmark:
    """
    Benchmarking framework for RAG strategies.

    Supports:
    - Multiple RAG strategies
    - Query categories (factual, multi-hop, temporal, etc.)
    - Difficulty levels
    - Performance metrics
    - Comparison reports
    """

    def __init__(
        self,
        silver_path: str,
        gold_path: str,
        pathrag_working_dir: str = "./data/pathrag_index",
        output_dir: str = "./data/benchmark_results"
    ):
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.pathrag_working_dir = pathrag_working_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded retrievers
        self._hybrid_retriever = None
        self._pathrag_pipeline = None

        # Results storage
        self.results: List[BenchmarkResult] = []

        logger.info(f"RAGBenchmark initialized")
        logger.info(f"  Silver: {silver_path}")
        logger.info(f"  Gold: {gold_path}")
        logger.info(f"  PathRAG: {pathrag_working_dir}")

    def _get_hybrid_retriever(self):
        """Get or create HybridRetriever."""
        if self._hybrid_retriever is None:
            from src.retrieval import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                str(self.gold_path),
                str(self.silver_path)
            )
        return self._hybrid_retriever

    def _get_pathrag_pipeline(self):
        """Get or create PathRAG pipeline."""
        if self._pathrag_pipeline is None:
            from src.benchmarks.pathrag_pipeline import PathRAGPipeline, PathRAGPipelineConfig
            config = PathRAGPipelineConfig(working_dir=self.pathrag_working_dir)
            self._pathrag_pipeline = PathRAGPipeline(str(self.silver_path), config)
        return self._pathrag_pipeline

    async def run_basic_rag(self, query: str) -> BenchmarkResult:
        """Run Basic RAG (vector search only)."""
        start_time = time.time()

        try:
            from src.retrieval import RetrievalStrategy
            retriever = self._get_hybrid_retriever()
            result = retriever.retrieve(query, RetrievalStrategy.VECTOR)

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.BASIC_RAG.value,
                answer=result.answer or "",
                execution_time=time.time() - start_time,
                chunks_retrieved=len(result.chunks),
                confidence=result.confidence,
                success=True,
                metadata=result.metadata
            )
        except Exception as e:
            logger.error(f"Basic RAG failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.BASIC_RAG.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_graphrag(self, query: str) -> BenchmarkResult:
        """Run GraphRAG (community-based)."""
        start_time = time.time()

        try:
            from src.retrieval import RetrievalStrategy
            retriever = self._get_hybrid_retriever()
            result = retriever.retrieve(query, RetrievalStrategy.GRAPHRAG)

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.GRAPHRAG.value,
                answer=result.answer or "",
                execution_time=time.time() - start_time,
                chunks_retrieved=len(result.chunks),
                confidence=result.confidence,
                success=True,
                metadata=result.metadata
            )
        except Exception as e:
            logger.error(f"GraphRAG failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.GRAPHRAG.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_pathrag(self, query: str) -> BenchmarkResult:
        """Run PathRAG (official implementation)."""
        start_time = time.time()

        try:
            pipeline = self._get_pathrag_pipeline()
            result = await pipeline.query(query, mode="hybrid")

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.PATHRAG.value,
                answer=result.get("answer", ""),
                execution_time=result.get("execution_time", time.time() - start_time),
                chunks_retrieved=0,  # PathRAG doesn't expose this directly
                confidence=1.0 if result.get("success") else 0.0,
                success=result.get("success", False),
                error=result.get("error"),
                metadata={"mode": result.get("mode")}
            )
        except Exception as e:
            logger.error(f"PathRAG failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.PATHRAG.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_react_graphrag(self, query: str) -> BenchmarkResult:
        """Run ReAct with GraphRAG tools."""
        start_time = time.time()

        try:
            from src.retrieval import RetrievalStrategy
            retriever = self._get_hybrid_retriever()
            result = retriever.retrieve(query, RetrievalStrategy.REACT)

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.REACT_GRAPHRAG.value,
                answer=result.answer or "",
                execution_time=time.time() - start_time,
                chunks_retrieved=len(result.chunks),
                confidence=result.confidence,
                success=True,
                metadata=result.metadata
            )
        except Exception as e:
            logger.error(f"ReAct+GraphRAG failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.REACT_GRAPHRAG.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_react_pathrag(self, query: str) -> BenchmarkResult:
        """Run ReAct with PathRAG tools."""
        start_time = time.time()

        try:
            from src.retrieval import RetrievalStrategy
            retriever = self._get_hybrid_retriever()
            # Use the REACT strategy — internally uses PathRAG as one of its tools
            result = retriever.retrieve(query, RetrievalStrategy.REACT)

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.REACT_PATHRAG.value,
                answer=result.answer or "",
                execution_time=time.time() - start_time,
                chunks_retrieved=len(result.chunks),
                confidence=result.confidence,
                success=True,
                metadata={**result.metadata, "variant": "pathrag"}
            )
        except Exception as e:
            logger.error(f"ReAct+PathRAG failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.REACT_PATHRAG.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_hybrid(self, query: str) -> BenchmarkResult:
        """Run Hybrid (combined approach)."""
        start_time = time.time()

        try:
            from src.retrieval import RetrievalStrategy
            retriever = self._get_hybrid_retriever()
            result = retriever.retrieve(query, RetrievalStrategy.HYBRID)

            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.HYBRID.value,
                answer=result.answer or "",
                execution_time=time.time() - start_time,
                chunks_retrieved=len(result.chunks),
                confidence=result.confidence,
                success=True,
                metadata=result.metadata
            )
        except Exception as e:
            logger.error(f"Hybrid failed: {e}")
            return BenchmarkResult(
                query=query,
                strategy=RAGStrategy.HYBRID.value,
                answer="",
                execution_time=time.time() - start_time,
                chunks_retrieved=0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def run_query(
        self,
        query: BenchmarkQuery,
        strategies: List[RAGStrategy]
    ) -> List[BenchmarkResult]:
        """Run a query against multiple strategies."""
        results = []

        for strategy in strategies:
            logger.info(f"Running {strategy.value} for: {query.query[:50]}...")

            if strategy == RAGStrategy.BASIC_RAG:
                result = await self.run_basic_rag(query.query)
            elif strategy == RAGStrategy.GRAPHRAG:
                result = await self.run_graphrag(query.query)
            elif strategy == RAGStrategy.PATHRAG:
                result = await self.run_pathrag(query.query)
            elif strategy == RAGStrategy.REACT_GRAPHRAG:
                result = await self.run_react_graphrag(query.query)
            elif strategy == RAGStrategy.REACT_PATHRAG:
                result = await self.run_react_pathrag(query.query)
            elif strategy == RAGStrategy.HYBRID:
                result = await self.run_hybrid(query.query)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
                continue

            # Add query metadata
            result.metadata["category"] = query.category
            result.metadata["difficulty"] = query.difficulty

            results.append(result)
            self.results.append(result)

            logger.info(f"  {strategy.value}: {result.execution_time:.2f}s, success={result.success}")

        return results

    async def run_benchmark(
        self,
        queries: List[BenchmarkQuery],
        strategies: Optional[List[RAGStrategy]] = None
    ) -> Dict[str, Any]:
        """
        Run full benchmark.

        Args:
            queries: List of benchmark queries
            strategies: Strategies to test (default: all)

        Returns:
            Benchmark results and summary
        """
        if strategies is None:
            strategies = [
                RAGStrategy.BASIC_RAG,
                RAGStrategy.GRAPHRAG,
                RAGStrategy.PATHRAG,
                RAGStrategy.HYBRID
            ]

        print("\n" + "=" * 60)
        print("RAG BENCHMARK")
        print("=" * 60)
        print(f"Queries: {len(queries)}")
        print(f"Strategies: {[s.value for s in strategies]}")
        print("=" * 60 + "\n")

        start_time = time.time()

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query.query[:60]}...")
            await self.run_query(query, strategies)

        total_time = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(strategies)

        # Save results
        self._save_results(queries, summary)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {self.output_dir}")

        return {
            "total_time": total_time,
            "total_queries": len(queries),
            "strategies": [s.value for s in strategies],
            "summary": summary
        }

    def _generate_summary(self, strategies: List[RAGStrategy]) -> Dict[str, BenchmarkSummary]:
        """Generate summary statistics for each strategy."""
        summaries = {}

        for strategy in strategies:
            strategy_results = [r for r in self.results if r.strategy == strategy.value]

            if not strategy_results:
                continue

            successful = [r for r in strategy_results if r.success]

            summary = BenchmarkSummary(
                strategy=strategy.value,
                total_queries=len(strategy_results),
                successful_queries=len(successful),
                avg_execution_time=sum(r.execution_time for r in strategy_results) / len(strategy_results),
                avg_chunks_retrieved=sum(r.chunks_retrieved for r in strategy_results) / len(strategy_results),
                avg_confidence=sum(r.confidence for r in successful) / len(successful) if successful else 0.0,
            )

            # Count by category and difficulty
            for result in strategy_results:
                cat = result.metadata.get("category", "unknown")
                diff = result.metadata.get("difficulty", "unknown")
                summary.queries_by_category[cat] = summary.queries_by_category.get(cat, 0) + 1
                summary.queries_by_difficulty[diff] = summary.queries_by_difficulty.get(diff, 0) + 1

            summaries[strategy.value] = summary

        return summaries

    def _save_results(self, queries: List[BenchmarkQuery], summary: Dict):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "queries": [asdict(q) for q in queries],
                "results": [r.to_dict() for r in self.results],
                "summary": {k: asdict(v) for k, v in summary.items()}
            }, f, indent=2, default=str)

        # Save comparison table
        comparison_file = self.output_dir / f"benchmark_comparison_{timestamp}.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# RAG Benchmark Results\n\n")
            f.write(f"**Date:** {timestamp}\n\n")

            f.write("## Summary\n\n")
            f.write("| Strategy | Success Rate | Avg Time (s) | Avg Confidence |\n")
            f.write("|----------|-------------|--------------|----------------|\n")

            for strategy, s in summary.items():
                success_rate = s.successful_queries / s.total_queries * 100 if s.total_queries > 0 else 0
                f.write(f"| {strategy} | {success_rate:.1f}% | {s.avg_execution_time:.2f} | {s.avg_confidence:.2f} |\n")

            f.write("\n## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### Query: {result.query[:80]}\n\n")
                f.write(f"**Strategy:** {result.strategy}\n\n")
                f.write(f"**Answer:** {result.answer[:500]}...\n\n")
                f.write(f"**Time:** {result.execution_time:.2f}s | **Confidence:** {result.confidence:.2f}\n\n")
                f.write("---\n\n")

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Comparison saved to {comparison_file}")

    def print_comparison(self):
        """Print comparison table to console."""
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Success':<10} {'Avg Time':<12} {'Avg Chunks':<12} {'Confidence':<12}")
        print("-" * 80)

        for strategy in RAGStrategy:
            results = [r for r in self.results if r.strategy == strategy.value]
            if not results:
                continue

            successful = len([r for r in results if r.success])
            avg_time = sum(r.execution_time for r in results) / len(results)
            avg_chunks = sum(r.chunks_retrieved for r in results) / len(results)
            avg_conf = sum(r.confidence for r in results) / len(results)

            print(f"{strategy.value:<20} {successful}/{len(results):<8} {avg_time:<12.2f} {avg_chunks:<12.1f} {avg_conf:<12.2f}")

        print("=" * 80)

    def pairwise_strategy_comparison(self) -> List[Dict[str, Any]]:
        """
        Generate pairwise comparisons between strategies.

        Compares: GraphRAG vs PathRAG, GraphRAG vs ReAct, PathRAG vs ReAct.
        For each pair, computes per-query differences in confidence and execution time.
        """
        PAIRS = [
            (RAGStrategy.GRAPHRAG, RAGStrategy.PATHRAG),
            (RAGStrategy.GRAPHRAG, RAGStrategy.REACT_GRAPHRAG),
            (RAGStrategy.PATHRAG, RAGStrategy.REACT_PATHRAG),
        ]

        comparisons = []

        for strat_a, strat_b in PAIRS:
            results_a = {r.query: r for r in self.results if r.strategy == strat_a.value and r.success}
            results_b = {r.query: r for r in self.results if r.strategy == strat_b.value and r.success}

            common_queries = set(results_a.keys()) & set(results_b.keys())
            if not common_queries:
                continue

            conf_diffs = []
            time_diffs = []
            for q in common_queries:
                conf_diffs.append(results_a[q].confidence - results_b[q].confidence)
                time_diffs.append(results_a[q].execution_time - results_b[q].execution_time)

            n = len(conf_diffs)
            avg_conf_diff = sum(conf_diffs) / n
            avg_time_diff = sum(time_diffs) / n

            comparison = {
                "strategy_a": strat_a.value,
                "strategy_b": strat_b.value,
                "num_queries": n,
                "avg_confidence_diff": round(avg_conf_diff, 4),
                "avg_time_diff_seconds": round(avg_time_diff, 3),
                "winner_confidence": strat_a.value if avg_conf_diff > 0 else strat_b.value,
                "winner_speed": strat_a.value if avg_time_diff < 0 else strat_b.value,
            }
            comparisons.append(comparison)

        return comparisons


class CrossModeBenchmark:
    """
    Benchmark the same queries across processing modes (LOCAL, LLM, HYBRID).

    Each mode has its own silver/gold directories. This class runs the same
    queries + strategies against each mode's data and produces a comparison.
    """

    MODES = ["local", "llm", "hybrid"]

    def __init__(
        self,
        data_root: str = "./data",
        pathrag_working_dir: str = "./data/pathrag_index",
        output_dir: str = "./data/benchmark_results",
    ):
        self.data_root = Path(data_root)
        self.pathrag_working_dir = pathrag_working_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # mode → RAGBenchmark
        self.mode_benchmarks: Dict[str, RAGBenchmark] = {}
        self._discover_modes()

    def _discover_modes(self):
        """Find which modes have both silver and gold directories."""
        for mode in self.MODES:
            silver = self.data_root / f"silver_{mode}"
            gold = self.data_root / f"gold_{mode}"
            if silver.exists() and gold.exists():
                self.mode_benchmarks[mode] = RAGBenchmark(
                    silver_path=str(silver),
                    gold_path=str(gold),
                    pathrag_working_dir=self.pathrag_working_dir,
                    output_dir=str(self.output_dir / mode),
                )
                logger.info(f"Discovered mode: {mode} (silver={silver}, gold={gold})")
            else:
                logger.info(f"Skipping mode: {mode} (silver={silver.exists()}, gold={gold.exists()})")

    async def run_cross_mode_benchmark(
        self,
        queries: List[BenchmarkQuery],
        strategies: Optional[List[RAGStrategy]] = None,
    ) -> Dict[str, Any]:
        """
        Run the same queries+strategies against every available mode.

        Returns combined report with per-mode results and cross-mode comparison.
        """
        if strategies is None:
            strategies = [
                RAGStrategy.BASIC_RAG,
                RAGStrategy.GRAPHRAG,
                RAGStrategy.PATHRAG,
                RAGStrategy.REACT_GRAPHRAG,
                RAGStrategy.HYBRID,
            ]

        if not self.mode_benchmarks:
            logger.error("No modes with both silver and gold directories found.")
            return {"error": "No modes available"}

        print("\n" + "=" * 70)
        print("CROSS-MODE RAG BENCHMARK")
        print("=" * 70)
        print(f"Modes:      {list(self.mode_benchmarks.keys())}")
        print(f"Queries:    {len(queries)}")
        print(f"Strategies: {[s.value for s in strategies]}")
        print("=" * 70)

        mode_reports = {}
        start_time = time.time()

        for mode, benchmark in self.mode_benchmarks.items():
            print(f"\n{'#' * 60}")
            print(f"# MODE: {mode.upper()}")
            print(f"{'#' * 60}")
            report = await benchmark.run_benchmark(queries, strategies)
            mode_reports[mode] = report

            # Pairwise strategy comparisons within this mode
            pw = benchmark.pairwise_strategy_comparison()
            mode_reports[mode]["pairwise_strategy_comparisons"] = pw

        total_time = time.time() - start_time

        # Generate cross-mode comparison
        cross_mode_report = self._generate_cross_mode_report(mode_reports, strategies)
        cross_mode_report["total_time"] = total_time

        # Save report
        self._save_cross_mode_report(cross_mode_report, mode_reports)

        # Print summary
        self._print_cross_mode_summary(cross_mode_report)

        return cross_mode_report

    def _generate_cross_mode_report(
        self,
        mode_reports: Dict[str, Any],
        strategies: List[RAGStrategy],
    ) -> Dict[str, Any]:
        """Build a structured cross-mode comparison."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "modes": list(mode_reports.keys()),
            "strategies": [s.value for s in strategies],
            "per_mode": {},
            "strategy_across_modes": {},
        }

        # Per-mode summaries
        for mode, mr in mode_reports.items():
            summary = mr.get("summary", {})
            report["per_mode"][mode] = {
                strat: {
                    "avg_confidence": getattr(s, "avg_confidence", 0),
                    "avg_execution_time": getattr(s, "avg_execution_time", 0),
                    "success_rate": (s.successful_queries / s.total_queries * 100
                                     if s.total_queries > 0 else 0),
                }
                for strat, s in summary.items()
            }

        # Strategy comparison across modes
        for strat in strategies:
            strat_name = strat.value
            mode_scores = {}
            for mode in mode_reports:
                summaries = mode_reports[mode].get("summary", {})
                if strat_name in summaries:
                    s = summaries[strat_name]
                    mode_scores[mode] = {
                        "avg_confidence": s.avg_confidence,
                        "avg_time": s.avg_execution_time,
                        "success_rate": (s.successful_queries / s.total_queries * 100
                                         if s.total_queries > 0 else 0),
                    }
            if mode_scores:
                best_mode = max(mode_scores, key=lambda m: mode_scores[m]["avg_confidence"])
                report["strategy_across_modes"][strat_name] = {
                    "by_mode": mode_scores,
                    "best_mode": best_mode,
                }

        # Pairwise strategy comparisons (aggregate across modes)
        report["pairwise_strategy_comparisons"] = {}
        for mode, mr in mode_reports.items():
            pw = mr.get("pairwise_strategy_comparisons", [])
            report["pairwise_strategy_comparisons"][mode] = pw

        return report

    def _save_cross_mode_report(self, report: Dict, mode_reports: Dict):
        """Save cross-mode benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_path = self.output_dir / f"cross_mode_benchmark_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Markdown report
        md_path = self.output_dir / f"cross_mode_benchmark_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Cross-Mode RAG Benchmark Report\n\n")
            f.write(f"**Generated:** {report['generated_at']}\n\n")
            f.write(f"**Modes:** {', '.join(report['modes'])}\n\n")

            # Per-strategy across-mode table
            f.write("## Strategy Performance Across Modes\n\n")
            modes = report["modes"]
            f.write(f"| Strategy | " + " | ".join(f"{m.upper()} Conf" for m in modes) + " | Best Mode |\n")
            f.write("|----------|" + "|".join(["--------"] * len(modes)) + "|-----------|\n")

            for strat_name, data in report.get("strategy_across_modes", {}).items():
                vals = []
                for m in modes:
                    conf = data["by_mode"].get(m, {}).get("avg_confidence", 0)
                    vals.append(f"{conf:.2f}")
                f.write(f"| {strat_name} | " + " | ".join(vals) + f" | {data['best_mode']} |\n")

            # Pairwise strategy comparisons per mode
            f.write("\n## Pairwise Strategy Comparisons\n\n")
            for mode, pairs in report.get("pairwise_strategy_comparisons", {}).items():
                f.write(f"### {mode.upper()}\n\n")
                if not pairs:
                    f.write("No pairwise data available.\n\n")
                    continue
                f.write("| Strategy A | Strategy B | Conf Diff | Speed Winner |\n")
                f.write("|------------|------------|-----------|-------------|\n")
                for p in pairs:
                    f.write(f"| {p['strategy_a']} | {p['strategy_b']} "
                            f"| {p['avg_confidence_diff']:+.4f} | {p['winner_speed']} |\n")
                f.write("\n")

        logger.info(f"Cross-mode report saved to {json_path} and {md_path}")

    def _print_cross_mode_summary(self, report: Dict):
        """Print cross-mode comparison summary to console."""
        print("\n" + "=" * 80)
        print("CROSS-MODE BENCHMARK SUMMARY")
        print("=" * 80)

        modes = report["modes"]
        header = f"{'Strategy':<20}" + "".join(f"{m.upper():>15}" for m in modes) + f"{'Best':>10}"
        print(header)
        print("-" * len(header))

        for strat_name, data in report.get("strategy_across_modes", {}).items():
            row = f"{strat_name:<20}"
            for m in modes:
                conf = data["by_mode"].get(m, {}).get("avg_confidence", 0)
                row += f"{conf:>15.2f}"
            row += f"{data['best_mode']:>10}"
            print(row)

        # Pairwise
        print("\n" + "-" * 80)
        print("PAIRWISE STRATEGY COMPARISONS")
        print("-" * 80)
        for mode, pairs in report.get("pairwise_strategy_comparisons", {}).items():
            print(f"\n  {mode.upper()}:")
            for p in pairs:
                diff = p["avg_confidence_diff"]
                arrow = ">" if diff > 0 else "<"
                print(f"    {p['strategy_a']} {arrow} {p['strategy_b']}  "
                      f"(conf diff: {diff:+.4f}, faster: {p['winner_speed']})")

        print("\n" + "=" * 80)


# Sample benchmark queries
SAMPLE_QUERIES = [
    BenchmarkQuery(
        query="What projects was Muhammad Rafiq working on?",
        category="factual",
        difficulty="easy",
        expected_entities=["Muhammad Rafiq", "OCR"],
    ),
    BenchmarkQuery(
        query="What did Rafiq and Imran work on together?",
        category="multi-hop",
        difficulty="medium",
        expected_entities=["Rafiq", "Imran"],
    ),
    BenchmarkQuery(
        query="Who discussed validation issues and what were the problems?",
        category="aggregation",
        difficulty="medium",
        expected_topics=["validation", "errors"],
    ),
    BenchmarkQuery(
        query="What were the main topics discussed in the email threads?",
        category="aggregation",
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Who organized the Iftar dinner and who participated?",
        category="multi-hop",
        difficulty="medium",
        expected_entities=["Iftar"],
    ),
]


def main():
    """CLI for benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Benchmarking — single-mode or cross-mode comparison"
    )

    # Mode selection
    parser.add_argument("--mode", "-m", choices=["local", "llm", "hybrid"],
                        help="Processing mode (auto-derives silver/gold paths)")
    parser.add_argument("--compare-modes", action="store_true",
                        help="Benchmark the same queries across ALL available modes")

    # Explicit paths (override --mode)
    parser.add_argument("--silver", help="Path to Silver layer (overrides --mode)")
    parser.add_argument("--gold", help="Path to Gold layer (overrides --mode)")

    parser.add_argument("--data-root", default="./data", help="Root data directory (default: ./data)")
    parser.add_argument("--pathrag-dir", default="./data/pathrag_index", help="PathRAG working directory")
    parser.add_argument("--output", default="./data/benchmark_results", help="Output directory")
    parser.add_argument("--strategies", nargs="+",
                        default=["basic_rag", "graphrag", "pathrag", "react_graphrag", "hybrid"],
                        choices=["basic_rag", "graphrag", "pathrag", "react_graphrag", "react_pathrag", "hybrid"],
                        help="Strategies to benchmark")
    parser.add_argument("--queries-file", help="JSON file with custom queries")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load queries
    if args.queries_file:
        with open(args.queries_file, 'r') as f:
            queries_data = json.load(f)
            queries = [BenchmarkQuery(**q) for q in queries_data]
    else:
        queries = SAMPLE_QUERIES

    strategies = [RAGStrategy(s) for s in args.strategies]

    loop = asyncio.new_event_loop()

    try:
        if args.compare_modes:
            # Cross-mode benchmark
            cross = CrossModeBenchmark(
                data_root=args.data_root,
                pathrag_working_dir=args.pathrag_dir,
                output_dir=args.output,
            )
            loop.run_until_complete(cross.run_cross_mode_benchmark(queries, strategies))
        else:
            # Single-mode benchmark
            data_root = Path(args.data_root)
            if args.silver and args.gold:
                silver, gold = args.silver, args.gold
            elif args.mode:
                silver = str(data_root / f"silver_{args.mode}")
                gold = str(data_root / f"gold_{args.mode}")
            else:
                parser.error("Provide --mode, --compare-modes, or both --silver and --gold")

            benchmark = RAGBenchmark(
                silver_path=silver,
                gold_path=gold,
                pathrag_working_dir=args.pathrag_dir,
                output_dir=args.output,
            )
            loop.run_until_complete(benchmark.run_benchmark(queries, strategies))
            benchmark.print_comparison()

            # Print pairwise strategy comparisons
            pw = benchmark.pairwise_strategy_comparison()
            if pw:
                print("\nPAIRWISE STRATEGY COMPARISONS:")
                print("-" * 60)
                for p in pw:
                    diff = p["avg_confidence_diff"]
                    print(f"  {p['strategy_a']} vs {p['strategy_b']}: "
                          f"conf diff={diff:+.4f}, faster={p['winner_speed']}")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
