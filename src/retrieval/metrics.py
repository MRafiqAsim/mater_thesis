"""
Retrieval Metrics Module
========================
Evaluation metrics for RAG system baseline measurement.

Metrics:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K
- Hit Rate
- Latency metrics

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    mrr: float  # Mean Reciprocal Rank
    mrr_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_10: float


@dataclass
class LatencyMetrics:
    """System latency metrics."""
    avg_retrieval_ms: float
    p50_retrieval_ms: float
    p95_retrieval_ms: float
    p99_retrieval_ms: float
    avg_generation_ms: float
    avg_total_ms: float


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    retrieval_metrics: RetrievalMetrics
    latency_metrics: LatencyMetrics
    num_queries: int
    configuration: Dict[str, Any]
    timestamp: str


class RetrievalEvaluator:
    """
    Evaluate retrieval quality against ground truth.

    Usage:
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(queries, ground_truth, retrieved_results)
    """

    @staticmethod
    def reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Calculate reciprocal rank.

        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (ordered by rank)

        Returns:
            Reciprocal rank (1/rank of first relevant result, 0 if none)
        """
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate DCG@K.

        Args:
            relevances: List of relevance scores (ordered by rank)
            k: Cutoff position

        Returns:
            DCG@K score
        """
        relevances = relevances[:k]
        if not relevances:
            return 0.0

        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / np.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate NDCG@K.

        Args:
            relevances: List of relevance scores (ordered by rank)
            k: Cutoff position

        Returns:
            NDCG@K score
        """
        dcg = RetrievalEvaluator.dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = RetrievalEvaluator.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        relevant_retrieved = len(retrieved_at_k & relevant_set)

        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_ids:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        relevant_retrieved = len(retrieved_at_k & relevant_set)

        return relevant_retrieved / len(relevant_set)

    @staticmethod
    def hit_rate_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """Calculate Hit Rate@K (binary: 1 if any relevant in top-k, else 0)."""
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return 1.0 if retrieved_at_k & relevant_set else 0.0

    def evaluate(
        self,
        queries: List[str],
        ground_truth: List[List[str]],  # List of relevant doc IDs per query
        retrieved_results: List[List[str]],  # List of retrieved doc IDs per query
        relevance_scores: Optional[List[List[float]]] = None  # Optional graded relevance
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.

        Args:
            queries: List of query strings
            ground_truth: List of relevant document IDs per query
            retrieved_results: List of retrieved document IDs per query
            relevance_scores: Optional graded relevance scores

        Returns:
            RetrievalMetrics
        """
        n = len(queries)
        if n == 0:
            raise ValueError("No queries to evaluate")

        # Calculate per-query metrics
        mrr_scores = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        precision_1_scores = []
        precision_5_scores = []
        precision_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        hit_1_scores = []
        hit_5_scores = []
        hit_10_scores = []

        for i in range(n):
            relevant = ground_truth[i]
            retrieved = retrieved_results[i]

            # MRR
            mrr_scores.append(self.reciprocal_rank(relevant, retrieved))

            # NDCG (using binary relevance if no scores provided)
            if relevance_scores and relevance_scores[i]:
                rels = relevance_scores[i]
            else:
                # Binary relevance: 1 if in ground truth, 0 otherwise
                relevant_set = set(relevant)
                rels = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved]

            ndcg_5_scores.append(self.ndcg_at_k(rels, 5))
            ndcg_10_scores.append(self.ndcg_at_k(rels, 10))

            # Precision
            precision_1_scores.append(self.precision_at_k(relevant, retrieved, 1))
            precision_5_scores.append(self.precision_at_k(relevant, retrieved, 5))
            precision_10_scores.append(self.precision_at_k(relevant, retrieved, 10))

            # Recall
            recall_5_scores.append(self.recall_at_k(relevant, retrieved, 5))
            recall_10_scores.append(self.recall_at_k(relevant, retrieved, 10))

            # Hit Rate
            hit_1_scores.append(self.hit_rate_at_k(relevant, retrieved, 1))
            hit_5_scores.append(self.hit_rate_at_k(relevant, retrieved, 5))
            hit_10_scores.append(self.hit_rate_at_k(relevant, retrieved, 10))

        return RetrievalMetrics(
            mrr=np.mean(mrr_scores),
            mrr_at_10=np.mean(mrr_scores),  # Same as MRR if retrieved >= 10
            ndcg_at_5=np.mean(ndcg_5_scores),
            ndcg_at_10=np.mean(ndcg_10_scores),
            precision_at_1=np.mean(precision_1_scores),
            precision_at_5=np.mean(precision_5_scores),
            precision_at_10=np.mean(precision_10_scores),
            recall_at_5=np.mean(recall_5_scores),
            recall_at_10=np.mean(recall_10_scores),
            hit_rate_at_1=np.mean(hit_1_scores),
            hit_rate_at_5=np.mean(hit_5_scores),
            hit_rate_at_10=np.mean(hit_10_scores),
        )


class LatencyEvaluator:
    """
    Evaluate system latency.
    """

    @staticmethod
    def evaluate(
        retrieval_times_ms: List[float],
        generation_times_ms: Optional[List[float]] = None
    ) -> LatencyMetrics:
        """
        Calculate latency metrics.

        Args:
            retrieval_times_ms: List of retrieval times in milliseconds
            generation_times_ms: Optional list of generation times

        Returns:
            LatencyMetrics
        """
        retrieval_arr = np.array(retrieval_times_ms)

        generation_avg = 0.0
        if generation_times_ms:
            generation_avg = np.mean(generation_times_ms)

        return LatencyMetrics(
            avg_retrieval_ms=np.mean(retrieval_arr),
            p50_retrieval_ms=np.percentile(retrieval_arr, 50),
            p95_retrieval_ms=np.percentile(retrieval_arr, 95),
            p99_retrieval_ms=np.percentile(retrieval_arr, 99),
            avg_generation_ms=generation_avg,
            avg_total_ms=np.mean(retrieval_arr) + generation_avg,
        )


class BaselineEvaluator:
    """
    Complete baseline evaluation for RAG system.

    Creates reproducible baseline measurements for comparison
    with GraphRAG and ReAct enhancements.
    """

    def __init__(self, rag_chain, test_queries: List[Dict[str, Any]]):
        """
        Initialize baseline evaluator.

        Args:
            rag_chain: RAGChain instance
            test_queries: List of test queries with ground truth
                         [{"query": "...", "relevant_docs": ["id1", "id2"]}]
        """
        self.rag_chain = rag_chain
        self.test_queries = test_queries
        self.retrieval_evaluator = RetrievalEvaluator()

    def run_evaluation(self) -> EvaluationResult:
        """
        Run complete baseline evaluation.

        Returns:
            EvaluationResult with all metrics
        """
        from datetime import datetime

        queries = []
        ground_truth = []
        retrieved_results = []
        retrieval_times = []
        generation_times = []

        for test_case in self.test_queries:
            query = test_case["query"]
            relevant_docs = test_case.get("relevant_docs", [])

            # Execute query
            response = self.rag_chain.query(query)

            # Collect results
            queries.append(query)
            ground_truth.append(relevant_docs)
            retrieved_results.append([s["chunk_id"] for s in response.sources])
            retrieval_times.append(response.retrieval_time_ms)
            generation_times.append(response.generation_time_ms)

        # Calculate metrics
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            queries, ground_truth, retrieved_results
        )

        latency_metrics = LatencyEvaluator.evaluate(
            retrieval_times, generation_times
        )

        return EvaluationResult(
            retrieval_metrics=retrieval_metrics,
            latency_metrics=latency_metrics,
            num_queries=len(queries),
            configuration={
                "top_k": self.rag_chain.config.top_k,
                "search_type": self.rag_chain.config.search_type,
                "model": self.rag_chain.config.model_deployment,
            },
            timestamp=datetime.utcnow().isoformat(),
        )

    def compare_search_types(self) -> Dict[str, RetrievalMetrics]:
        """
        Compare different search types (vector, keyword, hybrid).

        Returns:
            Dictionary mapping search type to metrics
        """
        results = {}

        for search_type in ["vector", "keyword", "hybrid"]:
            # Temporarily change search type
            original_type = self.rag_chain.config.search_type
            self.rag_chain.config.search_type = search_type
            self.rag_chain.retriever.search_type = search_type

            # Run evaluation
            eval_result = self.run_evaluation()
            results[search_type] = eval_result.retrieval_metrics

            # Restore original
            self.rag_chain.config.search_type = original_type
            self.rag_chain.retriever.search_type = original_type

        return results


def format_metrics_report(result: EvaluationResult) -> str:
    """Format evaluation result as readable report."""
    rm = result.retrieval_metrics
    lm = result.latency_metrics

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BASELINE EVALUATION REPORT                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Timestamp: {result.timestamp:<51} ║
║  Queries Evaluated: {result.num_queries:<43} ║
╠══════════════════════════════════════════════════════════════════╣
║  RETRIEVAL METRICS                                               ║
╠──────────────────────────────────────────────────────────────────╣
║  MRR@10              : {rm.mrr_at_10:.4f}                                    ║
║  NDCG@5              : {rm.ndcg_at_5:.4f}                                    ║
║  NDCG@10             : {rm.ndcg_at_10:.4f}                                    ║
║  Precision@1         : {rm.precision_at_1:.4f}                                    ║
║  Precision@5         : {rm.precision_at_5:.4f}                                    ║
║  Precision@10        : {rm.precision_at_10:.4f}                                    ║
║  Recall@5            : {rm.recall_at_5:.4f}                                    ║
║  Recall@10           : {rm.recall_at_10:.4f}                                    ║
║  Hit Rate@1          : {rm.hit_rate_at_1:.4f}                                    ║
║  Hit Rate@5          : {rm.hit_rate_at_5:.4f}                                    ║
║  Hit Rate@10         : {rm.hit_rate_at_10:.4f}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  LATENCY METRICS                                                 ║
╠──────────────────────────────────────────────────────────────────╣
║  Avg Retrieval       : {lm.avg_retrieval_ms:.1f} ms                                  ║
║  P50 Retrieval       : {lm.p50_retrieval_ms:.1f} ms                                  ║
║  P95 Retrieval       : {lm.p95_retrieval_ms:.1f} ms                                  ║
║  P99 Retrieval       : {lm.p99_retrieval_ms:.1f} ms                                  ║
║  Avg Generation      : {lm.avg_generation_ms:.1f} ms                                  ║
║  Avg Total           : {lm.avg_total_ms:.1f} ms                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  CONFIGURATION                                                   ║
║  Search Type: {result.configuration.get('search_type', 'N/A'):<49} ║
║  Top K: {result.configuration.get('top_k', 'N/A'):<56} ║
║  Model: {result.configuration.get('model', 'N/A'):<55} ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return report


# Export
__all__ = [
    'RetrievalEvaluator',
    'LatencyEvaluator',
    'BaselineEvaluator',
    'RetrievalMetrics',
    'LatencyMetrics',
    'EvaluationResult',
    'format_metrics_report',
]
