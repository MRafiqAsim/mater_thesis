"""
Evaluation Module
=================
Comprehensive evaluation framework for RAG systems.

Components:
- ragas_evaluator: RAGAS framework metrics
- comparative_analysis: Cross-system comparison
- report_generator: Evaluation report generation

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from .ragas_evaluator import (
    RAGASEvaluator,
    RAGASDatasetBuilder,
    RAGASConfig,
    EvaluationSample,
    EvaluationResult,
    AggregatedResults,
)

from .comparative_analysis import (
    ComparativeAnalyzer,
    PerformanceProfiler,
    SystemType,
    SystemResult,
    ComparisonResult,
    AggregatedComparison,
)

from .report_generator import (
    EvaluationReportGenerator,
    ReportConfig,
)

__all__ = [
    # RAGAS Evaluator
    'RAGASEvaluator',
    'RAGASDatasetBuilder',
    'RAGASConfig',
    'EvaluationSample',
    'EvaluationResult',
    'AggregatedResults',
    # Comparative Analysis
    'ComparativeAnalyzer',
    'PerformanceProfiler',
    'SystemType',
    'SystemResult',
    'ComparisonResult',
    'AggregatedComparison',
    # Report Generator
    'EvaluationReportGenerator',
    'ReportConfig',
]
