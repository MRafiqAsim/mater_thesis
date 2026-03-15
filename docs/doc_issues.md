# Document Processing Issues & Fixes

## 1. Encrypted Office Files (XLS, XLSX, DOCX)

**Error**: `Workbook is encrypted` / `File is not a zip file`

**Root Cause**: Microsoft Office default encryption — files saved with empty-password protection. Common in enterprise environments where Outlook or SharePoint auto-encrypts attachments.

**Fix**: `msoffcrypto-tool` decryption with empty password (`password=""`) as preprocessing step before parsing. Applied in `document_parser.py._decrypt_if_needed()`.

**Why empty password works**: Microsoft Office uses default encryption for metadata protection, not security. The actual content password is empty — the encryption wrapper is a legacy artifact.

---

## 2. Legacy Office Formats (PPT, DOC, XLS)

**Error**: `Cannot parse PPT file` / `BadZipFile: File is not a zip file`

**Root Cause**: Pre-2007 Office formats (`.ppt`, `.doc`, `.xls`) use OLE2 binary format, not the modern OOXML ZIP-based format (`.pptx`, `.docx`, `.xlsx`). Python libraries (`python-pptx`, `python-docx`, `openpyxl`) only support OOXML.

**Fix**: LibreOffice headless conversion — converts legacy formats to modern equivalents before parsing:
- `.ppt` → `.pptx`
- `.doc` → `.docx`
- `.xls` → `.xlsx`

Applied in `document_parser.py._convert_legacy()` using `subprocess` call to `libreoffice --headless --convert-to`.

**Prerequisite**: LibreOffice must be installed on the system (`apt install libreoffice-core` on Linux, `brew install libreoffice` on macOS).

---

## 3. Chart-Only Excel Sheets

**Error**: `'Chartsheet' object has no attribute 'iter_rows'`

**Root Cause**: Some Excel workbooks contain chart-only sheets (no cell data). `openpyxl` returns a `Chartsheet` object instead of a `Worksheet` — it has no `iter_rows()` method.

**Fix**: Skip chart sheets with `if not hasattr(sheet, 'iter_rows'): continue` in `_parse_xlsx()`.

---

## 4. Scanned/Image PDFs (Garbled Text)

**Error**: Extracted text contains `/g00/g01/g02` glyph patterns or empty strings.

**Root Cause**: PDFs created from scanned images have no text layer. `pypdf` extracts glyph IDs instead of actual characters, producing unreadable output.

**Fix**: PyMuPDF-based PDF classification (`_classify_pdf_page_fitz()`) with 5-signal per-page analysis:
1. Text block coverage (positioned text vs page area)
2. Image coverage (embedded images vs page area)
3. Font presence (named fonts in spans)
4. Character density (chars per page area)
5. Glyph pattern detection (`/gNN/` regex)

Classification result per page: `digital`, `scanned`, or `hybrid`. Scanned PDFs are routed to Vision OCR (GPT-4o) in LLM mode, or skipped in local mode.

---

## 5. RTF-Only Email Bodies (Pre-2003 Outlook)

**Error**: Empty `body_text` for emails from Outlook 2000/2003 era.

**Root Cause**: Pre-2003 Outlook stored email bodies exclusively in RTF format. The `plain_text_body` and `html_body` properties are empty — only `rtf_body` contains content.

**Fix**: RTF fallback in `pst_extractor.py` — if `body_text` is empty, try `message.rtf_body` and convert to plain text using `striprtf.rtf_to_text()`.

---

## 6. PST Attachment Size Detection

**Error**: `read_buffer()` fails with `TypeError` or returns empty content.

**Root Cause**: `pypff` attachment objects have inconsistent APIs across versions. Some expose `size` as a property, others as `get_size()` method. `read_buffer()` may require a size argument, or may not.

**Fix**: Multiple fallback chain in `pst_extractor.py`:
1. Try `att.size` property
2. Try `att.get_size()` method
3. Try `att.read_buffer(size)` with detected size
4. Try `att.read_buffer()` without arguments
5. Skip if all fail

---

## 7. Corrupted/Truncated Files

**Error**: Various — `BadZipFile`, `InvalidFileException`, `struct.error`

**Root Cause**: Email attachments can be corrupted during PST archival, incomplete downloads, or Outlook storage limits.

**Fix**: Graceful error handling — log warning with filename and continue processing. These files are tracked in `unsupported_docs/` for manual review. No crash, no data loss for other attachments.

---

## 8. DOC Files Without Antiword

**Error**: `UnpackError` or empty text from `.doc` files.

**Root Cause**: `textract` and `python-docx` cannot parse binary `.doc` format. The `antiword` system utility is the standard tool but may not be installed.

**Fix**: LibreOffice headless conversion (`.doc` → `.docx`) as primary path. `antiword` as optional fallback. Install: `apt install antiword` on Linux, `brew install antiword` on macOS.

---

## 9. Truncated Bronze JSON Files (Broken Email Serialization)

**Error**: `Expecting value: line 38 column 22 (char 1632)` — Silver pipeline warns and skips the file.

**Symptom**: Bronze email JSON files are truncated mid-field. The file ends at `"email_body_text":` with no value — valid JSON up to that point, then nothing. File size is smaller than expected.

**Root Cause**: Lone surrogate characters (U+D800–U+DFFF) in email body text. These appear when PST email bodies encoded in UTF-16 or mixed encodings are decoded through `latin-1` (which never raises an error but can produce invalid Unicode codepoints). The old code used `json.dump()` streaming directly to file with `ensure_ascii=False`. When the UTF-8 file writer hit a lone surrogate, it crashed — but the file handle had already flushed partial content to disk, leaving a truncated JSON file.

**Reproduction**: `json.dump({"a": "before", "body": "hello\ud800world"}, f, ensure_ascii=False)` writes `{"a": "before", "body": ` then crashes with `UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800'`. The file is left with 24 bytes of valid-looking but incomplete JSON.

**Fix (two layers)**:

1. **`pst_extractor.py` — `_decode_body()`**: Replaced manual sanitization with `ftfy.fix_text()` which handles all problematic characters in one call:
   - Lone surrogates (U+D800–U+DFFF) → replaced with `�`
   - Null bytes (`\x00`) → removed
   - C0 control characters → removed (tabs/newlines preserved)
   - Mojibake (e.g. `cafÃ©` → `café`) → fixed automatically
   - This is especially valuable for Dutch emails decoded with the wrong charset

2. **`bronze_loader.py` — `load_email()` / `load_document()`**: Changed from `json.dump(data, file)` to `json.dumps(data)` → `file.write(str)`. If serialization fails, the exception is raised *before* any file is created — no more truncated files on disk.

**Impact**: Existing broken files must be re-ingested (re-run Bronze pipeline). The Silver pipeline already gracefully skips broken JSON files with a warning — no crash or data loss.

---

## Summary of Dependencies Added

| Package | Purpose | Install |
|---------|---------|---------|
| `msoffcrypto-tool` | Decrypt empty-password Office files | `pip install msoffcrypto-tool` |
| `PyMuPDF` | PDF classification (digital/scanned/hybrid) | `pip install PyMuPDF` |
| `striprtf` | RTF to plain text conversion | `pip install striprtf` |
| `ftfy` | Fix broken Unicode, mojibake, surrogates | `pip install ftfy` |
| LibreOffice | Legacy Office format conversion | System package |
| antiword | DOC text extraction (optional) | System package |
