# Ingestion Pipeline Module
# Handles data extraction, parsing, and loading into Bronze layer

from .pst_extractor import PSTExtractor, EmailMessage
from .document_parser import DocumentParser, ParsedDocument
from .chunker import SemanticChunker, Chunk
from .language_detector import LanguageDetector
from .bronze_loader import BronzeLayerLoader
from .attachment_processor import AttachmentProcessor, AttachmentContent
from .attachment_classifier import AttachmentClassifier, ClassificationResult
from .email_sensitivity_classifier import EmailSensitivityClassifier, LLMSensitivityClassifier, SensitivityResult

__all__ = [
    "PSTExtractor",
    "EmailMessage",
    "DocumentParser",
    "ParsedDocument",
    "SemanticChunker",
    "Chunk",
    "LanguageDetector",
    "BronzeLayerLoader",
    "AttachmentProcessor",
    "AttachmentContent",
    "AttachmentClassifier",
    "ClassificationResult",
    "EmailSensitivityClassifier",
    "LLMSensitivityClassifier",
    "SensitivityResult",
]
