"""
Document Loader Factory
=======================
LangChain-based document loaders for heterogeneous file formats.

Supported formats:
- PDF (.pdf)
- Word (.docx, .doc)
- Excel (.xlsx, .xls, .xlsm)
- PowerPoint (.pptx, .ppt)
- Images (.jpg, .jpeg, .png) - OCR via Azure Document Intelligence

Author: Muhammad Rafiq
KU Leuven - Master Thesis
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """Result from document loading operation."""
    documents: List[Document]
    file_path: str
    file_type: str
    page_count: int
    char_count: int
    load_time_ms: float
    error: Optional[str] = None


class DocumentLoaderFactory:
    """
    Factory for loading documents from various file formats.

    Usage:
        factory = DocumentLoaderFactory()
        result = factory.load("/path/to/document.pdf")
        for doc in result.documents:
            print(doc.page_content)
    """

    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'word',
        '.doc': 'word',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xlsm': 'excel',
        '.pptx': 'powerpoint',
        '.ppt': 'powerpoint',
        '.txt': 'text',
        '.csv': 'csv',
        '.json': 'json',
    }

    def __init__(self, azure_doc_intelligence_endpoint: Optional[str] = None,
                 azure_doc_intelligence_key: Optional[str] = None):
        """
        Initialize the document loader factory.

        Args:
            azure_doc_intelligence_endpoint: Optional Azure Document Intelligence endpoint for OCR
            azure_doc_intelligence_key: Optional Azure Document Intelligence key
        """
        self.azure_di_endpoint = azure_doc_intelligence_endpoint
        self.azure_di_key = azure_doc_intelligence_key

    def load(self, file_path: str) -> LoaderResult:
        """
        Load a document from the given file path.

        Args:
            file_path: Path to the document file

        Returns:
            LoaderResult with documents and metadata
        """
        import time
        start_time = time.time()

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            return LoaderResult(
                documents=[],
                file_path=file_path,
                file_type="unknown",
                page_count=0,
                char_count=0,
                load_time_ms=0,
                error=f"Unsupported file extension: {ext}"
            )

        file_type = self.SUPPORTED_EXTENSIONS[ext]

        try:
            if file_type == 'pdf':
                documents = self._load_pdf(file_path)
            elif file_type == 'word':
                documents = self._load_word(file_path)
            elif file_type == 'excel':
                documents = self._load_excel(file_path)
            elif file_type == 'powerpoint':
                documents = self._load_powerpoint(file_path)
            elif file_type == 'text':
                documents = self._load_text(file_path)
            elif file_type == 'csv':
                documents = self._load_csv(file_path)
            elif file_type == 'json':
                documents = self._load_json(file_path)
            else:
                documents = []

            # Add common metadata
            documents = self._enrich_metadata(documents, file_path)

            load_time_ms = (time.time() - start_time) * 1000

            return LoaderResult(
                documents=documents,
                file_path=file_path,
                file_type=file_type,
                page_count=len(documents),
                char_count=sum(len(d.page_content) for d in documents),
                load_time_ms=load_time_ms
            )

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return LoaderResult(
                documents=[],
                file_path=file_path,
                file_type=file_type,
                page_count=0,
                char_count=0,
                load_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document."""
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_word(self, file_path: str) -> List[Document]:
        """Load Word document (.docx, .doc)."""
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel document."""
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        return loader.load()

    def _load_powerpoint(self, file_path: str) -> List[Document]:
        """Load PowerPoint document."""
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str) -> List[Document]:
        """Load plain text file."""
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file."""
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file_path)
        return loader.load()

    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON file."""
        from langchain_community.document_loaders import JSONLoader
        loader = JSONLoader(file_path, jq_schema='.', text_content=False)
        return loader.load()

    def _enrich_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """Add common metadata to all documents."""
        path = Path(file_path)
        file_stat = path.stat() if path.exists() else None

        common_metadata = {
            "source_file": str(path.name),
            "source_path": str(path),
            "file_extension": path.suffix.lower(),
            "file_size_bytes": file_stat.st_size if file_stat else 0,
            "modified_date": datetime.fromtimestamp(file_stat.st_mtime).isoformat() if file_stat else None,
            "document_hash": self._compute_hash(file_path),
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }

        for doc in documents:
            doc.metadata.update(common_metadata)

        return documents

    def _compute_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file for deduplication."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class BatchDocumentLoader:
    """
    Batch loader for processing multiple documents.

    Usage:
        batch_loader = BatchDocumentLoader()
        results = batch_loader.load_directory("/path/to/documents")
    """

    def __init__(self):
        self.factory = DocumentLoaderFactory()

    def load_directory(self, directory_path: str,
                       recursive: bool = True,
                       extensions: Optional[List[str]] = None) -> List[LoaderResult]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            extensions: List of extensions to process (default: all supported)

        Returns:
            List of LoaderResult objects
        """
        from pathlib import Path

        path = Path(directory_path)
        results = []

        if extensions is None:
            extensions = list(DocumentLoaderFactory.SUPPORTED_EXTENSIONS.keys())

        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for file_path in path.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    logger.info(f"Loading: {file_path}")
                    result = self.factory.load(str(file_path))
                    results.append(result)

        return results

    def load_files(self, file_paths: List[str]) -> List[LoaderResult]:
        """
        Load specific files.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of LoaderResult objects
        """
        results = []
        for file_path in file_paths:
            logger.info(f"Loading: {file_path}")
            result = self.factory.load(file_path)
            results.append(result)
        return results


# Export for use in notebooks
__all__ = ['DocumentLoaderFactory', 'BatchDocumentLoader', 'LoaderResult']
