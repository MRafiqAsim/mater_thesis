"""
SimpleRAG Configuration

Centralized configuration for the entire pipeline.
All AI operations record model name, version, and prompt version.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
    deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
    embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))

    # Model versioning for lineage
    model_name: str = "gpt-4o"
    model_version: str = "2024-02-15"
    prompt_version: str = "v1.0"


@dataclass
class AzureVisionConfig:
    """Azure Vision API configuration for OCR."""
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_VISION_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_VISION_API_KEY", ""))
    api_version: str = "2024-02-01"

    # Model versioning for lineage
    model_name: str = "azure-vision-ocr"
    model_version: str = "2024-02-01"


@dataclass
class DirectoryConfig:
    """Directory structure configuration."""
    base_dir: Path = field(default_factory=lambda: Path("./data/simplerag"))

    # Source monitoring
    source_folder: Path = field(default=None)
    inprogress_folder: Path = field(default=None)
    processed_folder: Path = field(default=None)
    error_folder: Path = field(default=None)

    # Layer directories
    bronze_dir: Path = field(default=None)
    silver_dir: Path = field(default=None)
    gold_dir: Path = field(default=None)

    # Silver staged sub-directories
    silver_chunks_dir: Path = field(default=None)
    silver_anonymized_dir: Path = field(default=None)
    silver_summarized_dir: Path = field(default=None)

    def __post_init__(self):
        """Initialize paths relative to base_dir."""
        if self.source_folder is None:
            self.source_folder = self.base_dir / "source"
        if self.inprogress_folder is None:
            self.inprogress_folder = self.base_dir / "inprogress"
        if self.processed_folder is None:
            self.processed_folder = self.base_dir / "processed"
        if self.error_folder is None:
            self.error_folder = self.base_dir / "error"
        if self.bronze_dir is None:
            self.bronze_dir = self.base_dir / "bronze"
        if self.silver_dir is None:
            self.silver_dir = self.base_dir / "silver"
        if self.gold_dir is None:
            self.gold_dir = self.base_dir / "gold"
        # Silver staged directories
        if self.silver_chunks_dir is None:
            self.silver_chunks_dir = self.silver_dir / "chunks"
        if self.silver_anonymized_dir is None:
            self.silver_anonymized_dir = self.silver_dir / "chunks_anonymized"
        if self.silver_summarized_dir is None:
            self.silver_summarized_dir = self.silver_dir / "chunks_summarized"

    def ensure_directories(self):
        """Create all required directories."""
        dirs = [
            self.source_folder,
            self.inprogress_folder,
            self.processed_folder,
            self.error_folder,
            self.bronze_dir,
            self.bronze_dir / "emails",
            self.bronze_dir / "documents",
            self.bronze_dir / "attachments",
            self.bronze_dir / "metadata",
            self.silver_dir,
            self.silver_chunks_dir,
            self.silver_anonymized_dir,
            self.silver_summarized_dir,
            self.silver_dir / "ocr",
            self.gold_dir,
            self.gold_dir / "embeddings",
            self.gold_dir / "index",
            self.gold_dir / "metadata"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class SimpleRAGConfig:
    """
    Main configuration for SimpleRAG pipeline.

    Follows strict medallion architecture rules:
    1. Bronze is immutable and append-only
    2. Silver enriches with AI (OCR, anonymization, summarization)
    3. Gold serves RAG queries
    4. All outputs preserve lineage to Bronze
    5. All AI operations record model info
    """

    # Sub-configurations
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    azure_vision: AzureVisionConfig = field(default_factory=AzureVisionConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)

    # Processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_batch_size: int = 32

    # Processing limits
    process_threshold: int = field(default_factory=lambda: int(os.getenv("PROCESS_THRESHOLD", "-1")))
    process_anonymization_summarization_together: bool = field(
        default_factory=lambda: os.getenv("PROCESS_ANONYMIZATION_SUMMARIZATION_TOGETHER", "false").lower() == "true"
    )

    # RAG settings
    top_k: int = 10
    similarity_threshold: float = 0.3  # Lower threshold for better recall

    # Supported file formats
    supported_formats: tuple = (
        ".pst", ".msg",  # Email
        ".pdf", ".docx", ".txt",  # Documents
        ".xlsx", ".pptx",  # Office
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp"  # Images
    )

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        if not self.azure_openai.endpoint:
            errors.append("AZURE_OPENAI_ENDPOINT not set")
        if not self.azure_openai.api_key:
            errors.append("AZURE_OPENAI_API_KEY not set")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True

    def initialize(self):
        """Initialize the pipeline directories."""
        self.directories.ensure_directories()
        self.validate()
