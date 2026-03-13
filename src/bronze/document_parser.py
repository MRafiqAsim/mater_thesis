"""
Document Parser

Parses various document formats (PDF, DOCX, XLSX, PPTX, TXT)
and extracts text content with metadata.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    CSV = "csv"
    RTF = "rtf"
    HTML = "html"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Parsed document with extracted content"""

    # Identification
    doc_id: str
    source_path: str
    doc_type: DocumentType

    # Content
    text: str
    pages: List[str] = field(default_factory=list)  # Text per page/sheet/slide
    tables: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: int = 0

    # Processing metadata
    extraction_time: datetime = field(default_factory=datetime.now)
    language: Optional[str] = None
    char_count: int = 0
    word_count: int = 0

    # Error tracking
    parse_errors: List[str] = field(default_factory=list)
    is_partially_parsed: bool = False

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "doc_type": self.doc_type.value,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "language": self.language,
            "extraction_time": self.extraction_time.isoformat(),
            "has_tables": len(self.tables) > 0,
            "parse_errors": self.parse_errors,
        }


class DocumentParser:
    """
    Parse various document formats and extract text.

    Supports:
    - PDF: Text and table extraction
    - DOCX/DOC: Full document parsing
    - XLSX/XLS: Spreadsheet content
    - PPTX/PPT: Slide content
    - TXT/CSV/RTF/HTML: Plain text
    """

    EXTENSION_MAP = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".doc": DocumentType.DOC,
        ".xlsx": DocumentType.XLSX,
        ".xls": DocumentType.XLS,
        ".pptx": DocumentType.PPTX,
        ".ppt": DocumentType.PPT,
        ".txt": DocumentType.TXT,
        ".csv": DocumentType.CSV,
        ".rtf": DocumentType.RTF,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
    }

    def __init__(
        self,
        extract_tables: bool = True,
        ocr_enabled: bool = False,
        max_file_size_mb: int = 100
    ):
        """
        Initialize the parser.

        Args:
            extract_tables: Whether to extract tables from documents
            ocr_enabled: Whether to use OCR for scanned PDFs
            max_file_size_mb: Maximum file size to process
        """
        self.extract_tables = extract_tables
        self.ocr_enabled = ocr_enabled
        self.max_file_size = max_file_size_mb * 1024 * 1024

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a document file.

        Args:
            file_path: Path to the document

        Returns:
            ParsedDocument with extracted content
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f} MB")

        # Determine document type
        doc_type = self._detect_type(path)

        # Generate document ID
        doc_id = self._generate_id(path)

        # Parse based on type
        parser_map = {
            DocumentType.PDF: self._parse_pdf,
            DocumentType.DOCX: self._parse_docx,
            DocumentType.DOC: self._parse_doc,
            DocumentType.XLSX: self._parse_xlsx,
            DocumentType.XLS: self._parse_xls,
            DocumentType.PPTX: self._parse_pptx,
            DocumentType.PPT: self._parse_ppt,
            DocumentType.TXT: self._parse_txt,
            DocumentType.CSV: self._parse_csv,
            DocumentType.RTF: self._parse_rtf,
            DocumentType.HTML: self._parse_html,
        }

        parser = parser_map.get(doc_type, self._parse_txt)

        try:
            result = parser(path)
            result.doc_id = doc_id
            result.source_path = str(path)
            result.doc_type = doc_type
            return result

        except Exception as e:
            logger.error(f"Error parsing {path}: {e}")
            # Return partial result
            return ParsedDocument(
                doc_id=doc_id,
                source_path=str(path),
                doc_type=doc_type,
                text="",
                parse_errors=[str(e)],
                is_partially_parsed=True
            )

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        doc_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """
        Parse document from bytes.

        Args:
            content: Document content as bytes
            filename: Original filename (for type detection)
            doc_type: Override document type

        Returns:
            ParsedDocument
        """
        import tempfile

        # Determine type from filename if not specified
        if doc_type is None:
            ext = Path(filename).suffix.lower()
            doc_type = self.EXTENSION_MAP.get(ext, DocumentType.UNKNOWN)

        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = self.parse(temp_path)
            result.source_path = filename  # Use original filename
            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _detect_type(self, path: Path) -> DocumentType:
        """Detect document type from extension"""
        ext = path.suffix.lower()
        return self.EXTENSION_MAP.get(ext, DocumentType.UNKNOWN)

    def _generate_id(self, path: Path) -> str:
        """Generate unique document ID"""
        stat = path.stat()
        unique_str = f"{path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    # =========================================================================
    # PDF Parser
    # =========================================================================

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        """Parse PDF document"""
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError("Install pypdf or PyPDF2: pip install pypdf")

        text_parts = []
        pages = []
        tables = []
        metadata = {}

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)

            # Extract metadata
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.get("/Title"),
                    "author": reader.metadata.get("/Author"),
                    "created": reader.metadata.get("/CreationDate"),
                    "modified": reader.metadata.get("/ModDate"),
                }

            # Extract text from each page
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                    text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {e}")
                    pages.append("")

        # Extract tables if enabled
        if self.extract_tables:
            tables = self._extract_pdf_tables(path)

        # Parse dates
        created_date = self._parse_pdf_date(metadata.get("created"))
        modified_date = self._parse_pdf_date(metadata.get("modified"))

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.PDF,
            text="\n\n".join(text_parts),
            pages=pages,
            tables=tables,
            title=metadata.get("title"),
            author=metadata.get("author"),
            created_date=created_date,
            modified_date=modified_date,
            page_count=len(pages),
        )

    def _extract_pdf_tables(self, path: Path) -> List[Dict]:
        """Extract tables from PDF using tabula or camelot"""
        tables = []

        try:
            import tabula
            dfs = tabula.read_pdf(str(path), pages='all', silent=True)

            for i, df in enumerate(dfs):
                tables.append({
                    "table_id": i,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "data": df.to_dict('records')
                })

        except ImportError:
            logger.debug("tabula not available, skipping table extraction")
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return tables

    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date format"""
        if not date_str:
            return None

        try:
            # Remove 'D:' prefix
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            if len(date_str) >= 14:
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:
                return datetime.strptime(date_str[:8], "%Y%m%d")
        except (ValueError, TypeError):
            pass

        return None

    # =========================================================================
    # DOCX Parser
    # =========================================================================

    def _parse_docx(self, path: Path) -> ParsedDocument:
        """Parse DOCX document"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

        doc = Document(str(path))
        text_parts = []
        tables = []

        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract tables
        if self.extract_tables:
            for i, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)

                if table_data:
                    tables.append({
                        "table_id": i,
                        "rows": len(table_data),
                        "data": table_data
                    })

                    # Also add table text to content
                    table_text = "\n".join(["\t".join(row) for row in table_data])
                    text_parts.append(f"\n[Table {i+1}]\n{table_text}")

        # Extract metadata
        props = doc.core_properties

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.DOCX,
            text="\n\n".join(text_parts),
            tables=tables,
            title=props.title,
            author=props.author,
            created_date=props.created,
            modified_date=props.modified,
            page_count=1,  # DOCX doesn't have reliable page count
        )

    def _parse_doc(self, path: Path) -> ParsedDocument:
        """Parse legacy DOC document.

        Tries in order:
        1. python-docx (works for DOCX-compatible DOC files)
        2. antiword (reliable for legacy binary DOC)

        antiword install: brew install antiword (mac) | apt install antiword (linux)
        """
        # 1. Try python-docx
        try:
            return self._parse_docx(path)
        except Exception:
            pass

        # 2. Try antiword
        try:
            import subprocess
            result = subprocess.run(
                ["antiword", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return ParsedDocument(
                    doc_id="",
                    source_path="",
                    doc_type=DocumentType.DOC,
                    text=result.stdout.strip(),
                )
        except FileNotFoundError:
            logger.warning("antiword not installed. Install: brew install antiword (mac) | apt install antiword (linux)")
        except Exception as e:
            logger.debug(f"antiword DOC parsing failed: {e}")

        raise RuntimeError(f"Cannot parse DOC file: {path.name}. Install antiword.")

    # =========================================================================
    # Excel Parser
    # =========================================================================

    def _parse_xlsx(self, path: Path) -> ParsedDocument:
        """Parse XLSX spreadsheet"""
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise ImportError("Install openpyxl: pip install openpyxl")

        wb = load_workbook(str(path), read_only=True, data_only=True)
        text_parts = []
        pages = []  # Each sheet as a "page"
        tables = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_text = [f"=== Sheet: {sheet_name} ==="]
            sheet_data = []

            for row in sheet.iter_rows():
                row_values = []
                for cell in row:
                    value = cell.value
                    if value is not None:
                        row_values.append(str(value))
                    else:
                        row_values.append("")

                if any(row_values):
                    sheet_text.append("\t".join(row_values))
                    sheet_data.append(row_values)

            page_text = "\n".join(sheet_text)
            pages.append(page_text)
            text_parts.append(page_text)

            if sheet_data:
                tables.append({
                    "sheet_name": sheet_name,
                    "rows": len(sheet_data),
                    "data": sheet_data
                })

        # Metadata
        props = wb.properties

        wb.close()

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.XLSX,
            text="\n\n".join(text_parts),
            pages=pages,
            tables=tables,
            title=props.title if props else None,
            author=props.creator if props else None,
            created_date=props.created if props else None,
            modified_date=props.modified if props else None,
            page_count=len(pages),
        )

    def _parse_xls(self, path: Path) -> ParsedDocument:
        """Parse legacy XLS spreadsheet"""
        try:
            import xlrd
        except ImportError:
            raise ImportError("Install xlrd: pip install xlrd")

        wb = xlrd.open_workbook(str(path))
        text_parts = []
        pages = []
        tables = []

        for sheet in wb.sheets():
            sheet_text = [f"=== Sheet: {sheet.name} ==="]
            sheet_data = []

            for row_idx in range(sheet.nrows):
                row_values = [str(sheet.cell_value(row_idx, col_idx))
                             for col_idx in range(sheet.ncols)]

                if any(row_values):
                    sheet_text.append("\t".join(row_values))
                    sheet_data.append(row_values)

            page_text = "\n".join(sheet_text)
            pages.append(page_text)
            text_parts.append(page_text)

            if sheet_data:
                tables.append({
                    "sheet_name": sheet.name,
                    "rows": len(sheet_data),
                    "data": sheet_data
                })

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.XLS,
            text="\n\n".join(text_parts),
            pages=pages,
            tables=tables,
            page_count=len(pages),
        )

    # =========================================================================
    # PowerPoint Parser
    # =========================================================================

    def _parse_pptx(self, path: Path) -> ParsedDocument:
        """Parse PPTX presentation"""
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("Install python-pptx: pip install python-pptx")

        prs = Presentation(str(path))
        text_parts = []
        pages = []  # Each slide as a "page"

        for i, slide in enumerate(prs.slides, 1):
            slide_text = [f"=== Slide {i} ==="]

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)

                # Extract table content
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        row_text = [cell.text for cell in row.cells]
                        slide_text.append("\t".join(row_text))

            page_text = "\n".join(slide_text)
            pages.append(page_text)
            text_parts.append(page_text)

        # Metadata
        props = prs.core_properties

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.PPTX,
            text="\n\n".join(text_parts),
            pages=pages,
            title=props.title,
            author=props.author,
            created_date=props.created,
            modified_date=props.modified,
            page_count=len(pages),
        )

    def _parse_ppt(self, path: Path) -> ParsedDocument:
        """Parse legacy PPT presentation"""
        # Try to use pptx (sometimes works) or convert
        try:
            return self._parse_pptx(path)
        except Exception:
            pass

        raise RuntimeError("Cannot parse PPT file. Convert to PPTX format.")

    # =========================================================================
    # Text-based Parsers
    # =========================================================================

    def _parse_txt(self, path: Path) -> ParsedDocument:
        """Parse plain text file"""
        # Try different encodings
        text = ""
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.TXT,
            text=text,
            page_count=1,
        )

    def _parse_csv(self, path: Path) -> ParsedDocument:
        """Parse CSV file"""
        import csv

        text_parts = []
        tables = []
        table_data = []

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                text_parts.append("\t".join(row))
                table_data.append(row)

        if table_data:
            tables.append({
                "rows": len(table_data),
                "columns": table_data[0] if table_data else [],
                "data": table_data
            })

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.CSV,
            text="\n".join(text_parts),
            tables=tables,
            page_count=1,
        )

    def _parse_rtf(self, path: Path) -> ParsedDocument:
        """Parse RTF file"""
        try:
            from striprtf.striprtf import rtf_to_text
            rtf_content = path.read_text(errors='replace')
            text = rtf_to_text(rtf_content)
        except ImportError:
            # Fallback: basic cleaning
            rtf_content = path.read_text(errors='replace')
            import re
            text = re.sub(r'\\[a-z]+\d*\s?', '', rtf_content)
            text = re.sub(r'[{}]', '', text)

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.RTF,
            text=text,
            page_count=1,
        )

    def _parse_html(self, path: Path) -> ParsedDocument:
        """Parse HTML file"""
        try:
            from bs4 import BeautifulSoup
            html = path.read_text(errors='replace')
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style
            for tag in soup(['script', 'style', 'head']):
                tag.decompose()

            text = soup.get_text(separator='\n')
            title = soup.title.string if soup.title else None

        except ImportError:
            # Fallback: regex cleaning
            import re
            html = path.read_text(errors='replace')
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text)
            title = None

        return ParsedDocument(
            doc_id="",
            source_path="",
            doc_type=DocumentType.HTML,
            text=text.strip(),
            title=title,
            page_count=1,
        )


# Convenience function
def parse_document(file_path: str) -> ParsedDocument:
    """
    Parse a document file.

    Args:
        file_path: Path to the document

    Returns:
        ParsedDocument with extracted content
    """
    parser = DocumentParser()
    return parser.parse(file_path)
