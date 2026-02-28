# Anonymization Module
# PII detection and anonymization for the Silver layer
#
# Supports three processing modes (configured via src/config.py):
# - OPENAI: Use OpenAI API for PII detection, anonymization, and summarization
# - LOCAL: Use local models (Presidio/spaCy/regex) - no summarization
# - HYBRID: Combine local for high-confidence, OpenAI for complex cases

from .pii_detector import PIIDetector, PIIEntity, PIIType
from .anonymizer import Anonymizer, AnonymizationResult
from .identity_registry import IdentityRegistry, Identity
from .silver_processor import SilverLayerProcessor
from .privacy_metrics import (
    PrivacyMetricsCalculator,
    PrivacyMetricsResult,
    TextPrivacyAnalyzer,
    AnonymizedRecord,
    QuasiIdentifier,
    SensitiveAttribute,
    EquivalenceClass,
    calculate_privacy_metrics,
    analyze_text_privacy,
)

# Lazy imports for OpenAI components (requires openai package)
def get_openai_detector():
    from .openai_pii_detector import OpenAIPIIDetector
    return OpenAIPIIDetector

def get_openai_anonymizer():
    from .openai_pii_detector import OpenAIAnonymizer
    return OpenAIAnonymizer

def get_openai_summarizer():
    from .openai_summarizer import OpenAISummarizer
    return OpenAISummarizer

def get_unified_processor():
    from .unified_processor import UnifiedProcessor
    return UnifiedProcessor

__all__ = [
    # PII Detection (Local)
    "PIIDetector",
    "PIIEntity",
    "PIIType",
    # Anonymization (Local)
    "Anonymizer",
    "AnonymizationResult",
    # Identity Registry
    "IdentityRegistry",
    "Identity",
    # Silver Layer Processing
    "SilverLayerProcessor",
    # Privacy Metrics
    "PrivacyMetricsCalculator",
    "PrivacyMetricsResult",
    "TextPrivacyAnalyzer",
    "AnonymizedRecord",
    "QuasiIdentifier",
    "SensitiveAttribute",
    "EquivalenceClass",
    "calculate_privacy_metrics",
    "analyze_text_privacy",
    # OpenAI Components (lazy loaded)
    "get_openai_detector",
    "get_openai_anonymizer",
    "get_openai_summarizer",
    "get_unified_processor",
]
