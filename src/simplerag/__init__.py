"""
SimpleRAG Pipeline - Clean Medallion Architecture Implementation

Following strict layer boundaries:
- Bronze: Raw ingestion, immutable, append-only, NO AI
- Silver: OCR, Anonymization, Summarization (Azure OpenAI/Vision)
- Gold: RAG retrieval with lineage preservation
"""

from .bronze.ingestion import BronzeIngestion
from .silver.processor import SilverProcessor
from .gold.rag_retriever import GoldRAGRetriever
from .utils.lineage import LineageTracker
from .utils.config import SimpleRAGConfig

__all__ = [
    'BronzeIngestion',
    'SilverProcessor',
    'GoldRAGRetriever',
    'LineageTracker',
    'SimpleRAGConfig'
]
