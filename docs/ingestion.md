# Ingestion Layer - Hands-On Implementation Guide

Build the document ingestion pipeline from scratch to understand every component.

**Target Platform:** Windows PC with VS Code and Python

---

## Table of Contents

1. [Create Project Structure](#1-create-project-structure)
2. [Set Up Python Environment](#2-set-up-python-environment)
3. [Create Files One by One](#3-create-files-one-by-one)
4. [Add Test Documents](#4-add-test-documents)
5. [Run the Pipeline](#5-run-the-pipeline)
6. [Check the Output](#6-check-the-output)
7. [Understanding the Code](#7-understanding-the-code)
8. [PST Email Processing](#8-pst-email-processing)
9. [Next Steps](#9-next-steps)

---

## 1. Create Project Structure

Open VS Code terminal (`Ctrl+``) and run:

```powershell
# Create project folder
mkdir mater_thesis
cd mater_thesis

# Create directory structure
mkdir src
mkdir src\ingestion
mkdir src\pipeline
mkdir data
mkdir data\input
mkdir data\input\documents
mkdir data\bronze
mkdir data\bronze\documents
mkdir data\bronze\metadata
```

---

## 2. Set Up Python Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\Activate

# Install basic dependencies
pip install python-docx PyPDF2 openpyxl python-dateutil
```

---

## 3. Create Files One by One

### File 1: `src/__init__.py`

Create empty file:

```powershell
New-Item -Path src\__init__.py -ItemType File
```

---

### File 2: `src/ingestion/__init__.py`

Create empty file:

```powershell
New-Item -Path src\ingestion\__init__.py -ItemType File
```

---

### File 3: `src/ingestion/document_parser.py`

Create this file and add the following content:

```python
"""
Document Parser - Extracts text from PDF, DOCX, XLSX, TXT files
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Represents a parsed document"""
    doc_id: str
    doc_type: DocumentType
    source_file: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[List[List[str]]] = field(default_factory=list)
    page_count: int = 0
    word_count: int = 0
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "source_file": self.source_file,
            "text": self.text,
            "metadata": self.metadata,
            "tables": self.tables,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
        }


class DocumentParser:
    """Parse various document formats and extract text"""

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document and extract text content.

        Args:
            file_path: Path to the document

        Returns:
            ParsedDocument with extracted content
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine document type
        doc_type = self._get_doc_type(path)

        # Generate document ID
        doc_id = self._generate_doc_id(path)

        # Parse based on type
        if doc_type == DocumentType.PDF:
            return self._parse_pdf(path, doc_id)
        elif doc_type == DocumentType.DOCX:
            return self._parse_docx(path, doc_id)
        elif doc_type == DocumentType.XLSX:
            return self._parse_xlsx(path, doc_id)
        elif doc_type == DocumentType.TXT:
            return self._parse_txt(path, doc_id)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _get_doc_type(self, path: Path) -> DocumentType:
        """Determine document type from extension"""
        ext = path.suffix.lower()
        mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".xlsx": DocumentType.XLSX,
            ".xls": DocumentType.XLSX,
            ".txt": DocumentType.TXT,
        }
        return mapping.get(ext, DocumentType.UNKNOWN)

    def _generate_doc_id(self, path: Path) -> str:
        """Generate unique document ID"""
        # Use filename + modification time for uniqueness
        stat = path.stat()
        unique_string = f"{path.name}_{stat.st_mtime}"
        hash_val = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        return f"DOC-{path.stem}-{hash_val}"

    def _parse_pdf(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse PDF document"""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = "\n\n".join(text_parts)

            return ParsedDocument(
                doc_id=doc_id,
                doc_type=DocumentType.PDF,
                source_file=str(path),
                text=text,
                page_count=len(reader.pages),
                word_count=len(text.split()),
                metadata=dict(reader.metadata) if reader.metadata else {},
            )

        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {e}")
            raise

    def _parse_docx(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse DOCX document"""
        try:
            from docx import Document

            doc = Document(str(path))

            # Extract paragraphs
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Extract tables if requested
            tables = []
            if self.extract_tables:
                for table in doc.tables:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text for cell in row.cells]
                        table_data.append(row_data)
                    tables.append(table_data)
                    # Also add table text to main text
                    for row in table_data:
                        text_parts.append(" | ".join(row))

            text = "\n".join(text_parts)

            return ParsedDocument(
                doc_id=doc_id,
                doc_type=DocumentType.DOCX,
                source_file=str(path),
                text=text,
                tables=tables,
                word_count=len(text.split()),
            )

        except Exception as e:
            logger.error(f"Error parsing DOCX {path}: {e}")
            raise

    def _parse_xlsx(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse XLSX spreadsheet"""
        try:
            from openpyxl import load_workbook

            wb = load_workbook(str(path), data_only=True)

            text_parts = []
            tables = []

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")

                table_data = []
                for row in sheet.iter_rows(values_only=True):
                    # Filter None values and convert to strings
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    if any(row_values):  # Skip empty rows
                        table_data.append(row_values)
                        text_parts.append(" | ".join(row_values))

                if table_data:
                    tables.append(table_data)

            text = "\n".join(text_parts)

            return ParsedDocument(
                doc_id=doc_id,
                doc_type=DocumentType.XLSX,
                source_file=str(path),
                text=text,
                tables=tables,
                word_count=len(text.split()),
            )

        except Exception as e:
            logger.error(f"Error parsing XLSX {path}: {e}")
            raise

    def _parse_txt(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse plain text file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text = None

            for encoding in encodings:
                try:
                    text = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise ValueError(f"Could not decode {path}")

            return ParsedDocument(
                doc_id=doc_id,
                doc_type=DocumentType.TXT,
                source_file=str(path),
                text=text,
                word_count=len(text.split()),
            )

        except Exception as e:
            logger.error(f"Error parsing TXT {path}: {e}")
            raise
```

---

### File 4: `src/ingestion/bronze_loader.py`

Create this file and add the following content:

```python
"""
Bronze Layer Loader - Saves parsed documents to Bronze layer
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator

from .document_parser import ParsedDocument

logger = logging.getLogger(__name__)


class BronzeLayerLoader:
    """Load raw data into the Bronze layer"""

    def __init__(self, bronze_path: str):
        """
        Initialize the Bronze layer loader.

        Args:
            bronze_path: Base path for Bronze layer
        """
        self.bronze_path = Path(bronze_path)
        self._create_directories()

        # Statistics
        self.stats = {
            "documents_loaded": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat(),
        }

    def _create_directories(self) -> None:
        """Create Bronze layer directory structure"""
        directories = [
            self.bronze_path / "documents" / "pdf",
            self.bronze_path / "documents" / "docx",
            self.bronze_path / "documents" / "xlsx",
            self.bronze_path / "documents" / "txt",
            self.bronze_path / "documents" / "other",
            self.bronze_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_document(self, document: ParsedDocument) -> str:
        """
        Load a parsed document into the Bronze layer.

        Args:
            document: ParsedDocument to load

        Returns:
            Path where document was saved
        """
        try:
            # Determine subdirectory based on type
            doc_type = document.doc_type.value
            doc_dir = self.bronze_path / "documents" / doc_type

            if not doc_dir.exists():
                doc_dir = self.bronze_path / "documents" / "other"

            doc_dir.mkdir(parents=True, exist_ok=True)

            # Save document as JSON
            doc_path = doc_dir / f"{document.doc_id}.json"

            doc_data = document.to_dict()

            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False, default=str)

            self.stats["documents_loaded"] += 1
            logger.info(f"Loaded: {document.doc_id} -> {doc_path}")

            return str(doc_path)

        except Exception as e:
            logger.error(f"Error loading document {document.doc_id}: {e}")
            self.stats["errors"] += 1
            raise

    def save_metadata(self) -> str:
        """Save ingestion metadata and statistics"""
        self.stats["end_time"] = datetime.now().isoformat()

        metadata_path = self.bronze_path / "metadata" / "ingestion_log.json"

        # Load existing logs if present
        existing_logs = []
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    existing_logs = json.load(f)
            except Exception:
                existing_logs = []

        existing_logs.append(self.stats)

        with open(metadata_path, "w") as f:
            json.dump(existing_logs, f, indent=2, default=str)

        return str(metadata_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()

    def iter_documents(self) -> Iterator[Dict[str, Any]]:
        """Iterate over documents in Bronze layer"""
        docs_dir = self.bronze_path / "documents"

        for json_file in docs_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
```

---

### File 5: `src/pipeline/__init__.py`

Create empty file:

```powershell
New-Item -Path src\pipeline\__init__.py -ItemType File
```

---

### File 6: `src/pipeline/run_ingestion.py`

Create this file and add the following content:

```python
"""
Simple Ingestion Pipeline - Parse documents to Bronze layer
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.document_parser import DocumentParser
from ingestion.bronze_loader import BronzeLayerLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ingestion(input_dir: str, output_dir: str):
    """
    Run the ingestion pipeline.

    Args:
        input_dir: Directory containing source documents
        output_dir: Output directory for Bronze layer
    """
    print("\n" + "=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    input_path = Path(input_dir)
    bronze_path = Path(output_dir) / "bronze"

    # Supported extensions
    extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt"]

    # Find all documents
    doc_files = []
    for ext in extensions:
        doc_files.extend(input_path.rglob(f"*{ext}"))

    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {bronze_path}")
    print(f"Documents found: {len(doc_files)}")

    if not doc_files:
        print("\nNo documents found! Add files to:", input_path)
        return

    # Initialize parser and loader
    parser = DocumentParser(extract_tables=True)
    loader = BronzeLayerLoader(bronze_path=str(bronze_path))

    # Process each document
    print("\nProcessing documents...")
    print("-" * 40)

    for i, doc_path in enumerate(doc_files, 1):
        try:
            print(f"[{i}/{len(doc_files)}] {doc_path.name}...", end=" ")

            # Parse document
            document = parser.parse(str(doc_path))

            # Load to Bronze layer
            loader.load_document(document)

            print(f"OK ({document.word_count} words)")

        except Exception as e:
            print(f"FAILED: {e}")

    # Save metadata
    loader.save_metadata()

    # Print summary
    stats = loader.get_stats()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {stats['documents_loaded']}")
    print(f"Errors: {stats['errors']}")
    print(f"Bronze layer: {bronze_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run document ingestion")
    parser.add_argument(
        "--input", "-i",
        default="data/input/documents",
        help="Input directory with documents"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory"
    )

    args = parser.parse_args()

    run_ingestion(args.input, args.output)
```

---

## 4. Add Test Documents

Create a sample text file to test:

```powershell
@"
Meeting Notes - Project Alpha
Date: January 15, 2024

Attendees:
- John Smith (john.smith@company.com)
- Sarah Johnson (sarah.j@company.com)

Discussion:
We discussed the Q1 budget of $150,000 for the project.
John's phone number is +1 (555) 123-4567.

Action Items:
1. Sarah to prepare the proposal by January 20
2. John to contact the vendor

Next meeting: January 22, 2024
"@ | Out-File -FilePath data\input\documents\meeting_notes.txt -Encoding UTF8
```

Or manually create `data\input\documents\meeting_notes.txt` with similar content.

---

## 5. Run the Pipeline

```powershell
# Make sure virtual environment is active
.\.venv\Scripts\Activate

# Run ingestion
python src\pipeline\run_ingestion.py --input data\input\documents --output data
```

Expected output:

```
============================================================
DOCUMENT INGESTION PIPELINE
============================================================

Input directory: data\input\documents
Output directory: data\bronze
Documents found: 1

Processing documents...
----------------------------------------
[1/1] meeting_notes.txt... OK (45 words)

============================================================
INGESTION COMPLETE
============================================================
Documents processed: 1
Errors: 0
Bronze layer: data\bronze
============================================================
```

---

## 6. Check the Output

```powershell
# View the Bronze layer output
Get-Content data\bronze\documents\txt\*.json
```

Expected JSON structure:

```json
{
  "doc_id": "DOC-meeting_notes-a1b2c3d4",
  "doc_type": "txt",
  "source_file": "data\\input\\documents\\meeting_notes.txt",
  "text": "Meeting Notes - Project Alpha\nDate: January 15, 2024\n...",
  "metadata": {},
  "tables": [],
  "page_count": 0,
  "word_count": 45,
  "created_date": null,
  "modified_date": null
}
```

---

## 7. Understanding the Code

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Files    │────▶│  DocumentParser  │────▶│  BronzeLoader   │
│  (PDF/DOCX/TXT) │     │  (Extract text)  │     │  (Save JSON)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Bronze Layer   │
                                                 │  (JSON files)   │
                                                 └─────────────────┘
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `DocumentParser` | `document_parser.py` | Extracts text from different file formats |
| `ParsedDocument` | `document_parser.py` | Data class holding parsed content |
| `BronzeLayerLoader` | `bronze_loader.py` | Saves documents to Bronze layer as JSON |

### Document ID Generation

```python
# Unique ID = filename + modification time hash
doc_id = f"DOC-{filename}-{hash[:8]}"
# Example: DOC-meeting_notes-a1b2c3d4
```

This ensures:
- Same file → Same ID (idempotent)
- Different files → Different IDs
- Modified file → New ID

---

## 8. PST Email Processing

Process Outlook PST files to extract emails into the Bronze layer.

### 8.1 Install PST Library

```powershell
pip install libpff-python
```

> **Note:** On Windows, if installation fails, you may need to use WSL or find a pre-built wheel.

### 8.2 Create PST Extractor

Create `src/ingestion/pst_extractor.py`:

```python
"""
PST Extractor - Extracts emails from Outlook PST files
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)


@dataclass
class EmailAttachment:
    """Email attachment"""
    filename: str
    size: int
    content: bytes = field(repr=False)


@dataclass
class EmailMessage:
    """Represents an extracted email"""
    message_id: str
    subject: str
    sender: str
    recipients_to: List[str]
    recipients_cc: List[str]
    sent_time: Optional[datetime]
    body_text: str
    body_html: str
    attachments: List[EmailAttachment] = field(default_factory=list)
    source_pst: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "subject": self.subject,
            "sender": self.sender,
            "recipients_to": self.recipients_to,
            "recipients_cc": self.recipients_cc,
            "sent_time": self.sent_time.isoformat() if self.sent_time else None,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "has_attachments": len(self.attachments) > 0,
            "attachment_count": len(self.attachments),
            "attachment_names": [a.filename for a in self.attachments],
            "source_pst": self.source_pst,
        }


class PSTExtractor:
    """Extract emails from PST files"""

    def __init__(self, extract_attachments: bool = True):
        self.extract_attachments = extract_attachments
        self._check_library()

    def _check_library(self):
        """Check if pypff is available"""
        try:
            import pypff
            self.pypff = pypff
            logger.info("Using pypff library")
        except ImportError:
            raise ImportError(
                "pypff not installed. Try: pip install libpff-python\n"
                "On Windows, you may need to use WSL or a pre-built wheel."
            )

    def extract(self, pst_path: str) -> Iterator[EmailMessage]:
        """
        Extract emails from a PST file.

        Args:
            pst_path: Path to the PST file

        Yields:
            EmailMessage objects
        """
        path = Path(pst_path)
        if not path.exists():
            raise FileNotFoundError(f"PST file not found: {pst_path}")

        logger.info(f"Opening PST: {pst_path}")

        pst = self.pypff.file()
        pst.open(str(path))

        try:
            root = pst.get_root_folder()
            yield from self._process_folder(root, str(path))
        finally:
            pst.close()

    def _process_folder(self, folder, pst_path: str) -> Iterator[EmailMessage]:
        """Process a folder and its subfolders"""
        # Process messages in this folder
        for i in range(folder.get_number_of_sub_messages()):
            try:
                message = folder.get_sub_message(i)
                email = self._extract_message(message, pst_path)
                if email:
                    yield email
            except Exception as e:
                logger.warning(f"Error extracting message: {e}")

        # Process subfolders recursively
        for i in range(folder.get_number_of_sub_folders()):
            subfolder = folder.get_sub_folder(i)
            yield from self._process_folder(subfolder, pst_path)

    def _extract_message(self, message, pst_path: str) -> Optional[EmailMessage]:
        """Extract a single message"""
        try:
            # Generate message ID
            subject = message.get_subject() or ""
            sent_time = message.get_delivery_time()

            unique_str = f"{subject}_{sent_time}_{hash(message)}"
            msg_id = f"MSG-{hashlib.md5(unique_str.encode()).hexdigest()[:12]}"

            # Extract recipients
            recipients_to = []
            recipients_cc = []

            # Get body
            body_text = message.get_plain_text_body() or ""
            if isinstance(body_text, bytes):
                body_text = body_text.decode('utf-8', errors='ignore')

            body_html = message.get_html_body() or ""
            if isinstance(body_html, bytes):
                body_html = body_html.decode('utf-8', errors='ignore')

            # Extract attachments
            attachments = []
            if self.extract_attachments:
                for j in range(message.get_number_of_attachments()):
                    try:
                        att = message.get_attachment(j)
                        attachments.append(EmailAttachment(
                            filename=att.get_name() or f"attachment_{j}",
                            size=att.get_size(),
                            content=att.read_buffer(att.get_size()) if att.get_size() > 0 else b""
                        ))
                    except Exception as e:
                        logger.warning(f"Error extracting attachment: {e}")

            return EmailMessage(
                message_id=msg_id,
                subject=subject,
                sender=message.get_sender_name() or "",
                recipients_to=recipients_to,
                recipients_cc=recipients_cc,
                sent_time=sent_time,
                body_text=body_text,
                body_html=body_html,
                attachments=attachments,
                source_pst=pst_path,
            )

        except Exception as e:
            logger.error(f"Error extracting message: {e}")
            return None
```

### 8.3 Update Bronze Loader for Emails

Add this method to `src/ingestion/bronze_loader.py`:

```python
def load_email(self, email) -> str:
    """Load an email into Bronze layer"""
    try:
        # Determine path based on date
        if email.sent_time:
            year = email.sent_time.year
            month = f"{email.sent_time.month:02d}"
        else:
            year = "unknown"
            month = "unknown"

        email_dir = self.bronze_path / "emails" / str(year) / month
        email_dir.mkdir(parents=True, exist_ok=True)

        # Save email as JSON
        email_path = email_dir / f"{email.message_id}.json"

        with open(email_path, "w", encoding="utf-8") as f:
            json.dump(email.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        self.stats["documents_loaded"] += 1
        return str(email_path)

    except Exception as e:
        logger.error(f"Error loading email: {e}")
        self.stats["errors"] += 1
        raise
```

### 8.4 Create PST Ingestion Script

Create `src/pipeline/run_pst_ingestion.py`:

```python
"""
PST Ingestion Pipeline
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pst_extractor import PSTExtractor
from ingestion.bronze_loader import BronzeLayerLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pst_ingestion(pst_path: str, output_dir: str):
    """Run PST ingestion"""
    print("\n" + "=" * 60)
    print("PST EMAIL INGESTION")
    print("=" * 60)

    pst_file = Path(pst_path)
    if not pst_file.exists():
        print(f"ERROR: PST file not found: {pst_path}")
        return

    bronze_path = Path(output_dir) / "bronze"

    print(f"\nPST file: {pst_file}")
    print(f"Output: {bronze_path}")

    # Initialize
    extractor = PSTExtractor(extract_attachments=True)
    loader = BronzeLayerLoader(bronze_path=str(bronze_path))

    # Process
    print("\nExtracting emails...")
    print("-" * 40)

    count = 0
    for email in extractor.extract(str(pst_file)):
        count += 1
        subject = email.subject[:50] if email.subject else "(no subject)"
        print(f"[{count}] {subject}...")
        loader.load_email(email)

    loader.save_metadata()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Emails extracted: {count}")
    print(f"Bronze layer: {bronze_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pst", "-p", required=True, help="Path to PST file")
    parser.add_argument("--output", "-o", default="data", help="Output directory")

    args = parser.parse_args()
    run_pst_ingestion(args.pst, args.output)
```

### 8.5 Place Your PST File

```powershell
# Create PST input directory
mkdir data\input\pst

# Copy your PST file
copy C:\path\to\your\emails.pst data\input\pst\
```

### 8.6 Run PST Ingestion

```powershell
# Activate environment
.\.venv\Scripts\Activate

# Run PST ingestion only (Bronze layer)
python src\pipeline\run_pst_ingestion.py --pst data\input\pst\your_file.pst --output data
```

### 8.7 Check Extracted Emails

```powershell
# Count extracted emails
(Get-ChildItem -Path data\bronze\emails -Recurse -Filter "*.json").Count

# View sample email
Get-Content (Get-ChildItem data\bronze\emails\*\*\*.json | Select-Object -First 1)
```

### 8.8 Sample Output

**Bronze Layer Structure:**
```
data/bronze/
├── emails/
│   ├── 2013/
│   │   ├── 01/
│   │   │   ├── MSG-abc123.json
│   │   │   └── MSG-def456.json
│   │   └── 02/
│   ├── 2014/
│   └── unknown/
└── metadata/
    └── ingestion_log.json
```

**Sample Email JSON:**
```json
{
  "message_id": "MSG-f1f7697933f7",
  "subject": "RE: Project Update",
  "sender": "John Smith",
  "recipients_to": [],
  "sent_time": "2013-05-24T10:06:54",
  "body_text": "Please review the attached document...",
  "has_attachments": false,
  "source_pst": "data/input/pst/emails.pst"
}
```

### 8.9 Full Pipeline (PST + Anonymization)

To run the complete pipeline including Silver layer processing:

```powershell
# Full pipeline with anonymization (requires additional dependencies)
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg

# Run full pipeline
python src\pipeline\run_ingestion.py --pst data\input\pst\your_file.pst --output data
```

This will:
1. Extract emails to Bronze layer
2. Chunk and anonymize text
3. Save to Silver layer with PII removed

---

## 9. Next Steps

### Project Structure After Completion

```
mater_thesis/
├── .venv/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_parser.py
│   │   └── bronze_loader.py
│   └── pipeline/
│       ├── __init__.py
│       └── run_ingestion.py
└── data/
    ├── input/
    │   └── documents/
    │       └── meeting_notes.txt
    └── bronze/
        ├── documents/
        │   └── txt/
        │       └── DOC-meeting_notes-xxxxx.json
        └── metadata/
            └── ingestion_log.json
```

### Continue Building

1. **Add Chunking** - Split large documents into smaller chunks
2. **Add Anonymization** - Detect and mask PII
3. **Add Silver Layer** - Process Bronze → Silver with anonymization

---

## Quick Reference

```powershell
# Activate environment
.\.venv\Scripts\Activate

# Run ingestion
python src\pipeline\run_ingestion.py -i data\input\documents -o data

# Check output
Get-Content data\bronze\documents\txt\*.json

# View ingestion log
Get-Content data\bronze\metadata\ingestion_log.json
```
