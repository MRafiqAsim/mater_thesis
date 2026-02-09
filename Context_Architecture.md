# Project Context & Architecture Requirements

## Project Context

I have completed a thesis containing a literature review (Attached with the name of Master_Thesis.pdf) and now have the data to implement the project.

### Dataset Characteristics

- ~10-15 GB total size
- PST email archives spanning ~35 years
- PDF files folders
- MS Word docx, doc files
- Excel files (XLS, XLSX, XLSM)
- PPT files
- Emails contain attachments and cross-references to other documents
- Other File formats include:
  - MSG
  - JPG
  - DWG
- Multilingual communication: English and Dutch

### Functional Requirements

- Large-scale email archive and Documents (pdf, doc, docx, xlsx, xls etc) ingestion
- Email parsing, threading, and metadata extraction
- Attachment extraction and document linking
- Cross-document reference resolution
- Multilingual NLP processing
- Summarization and anonymization using Azure OpenAI

### Non-Functional Requirements

- Fully Azure-based deployment
- Scalable, fault-tolerant architecture (microservices, containers etc whatever is best)
- Industry-grade security and compliance
- Reproducible results suitable for academic evaluation

The solution must explicitly address 7 key challenges listed in the Master_Thesis.pdf, step by step, from ingestion through evaluation.

---

## Architecture Requirements

Using the project context above, write a state-of-the-art, Azure-native scalable project architecture document.

The architecture must:

- Be 100% deployed on Microsoft Azure
- Support large historical PST email and documents archives
- Handle heterogeneous document formats
- Support multilingual NLP (English & Dutch)
- Integrate Azure OpenAI for summarization and anonymization

### Document Structure

1. Architectural objectives and constraints
2. High-level system overview
3. Logical architecture (major components)
4. End-to-end data flow (step-by-step)
5. Storage architecture (raw, processed, curated zones)
6. Compute & orchestration architecture
7. NLP & LLM integration architecture
8. Security, compliance, and data governance
9. Scalability and performance strategy
10. Cost optimization and quota management
11. Azure service mapping (explicit service names)

Include text-based architecture diagrams and provide clear rationale for each design decision.
