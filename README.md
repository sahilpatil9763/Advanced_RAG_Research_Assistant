# Atlas Research — Advanced RAG Research Assistant

A fresh, accessible research workspace for uploading multiple PDFs, indexing their content, searching across documents, and asking grounded questions with source filenames and page citations.

This project keeps the current stack (Streamlit, PyMuPDF, Sentence Transformers, ChromaDB, and Groq) but uses a new dashboard design: a calm light theme, lavender accents, clear navigation, readable document cards, quick workspace metrics, and dedicated areas for chat, the document library, research tools, and settings.

## Features in this milestone

- Upload multiple text-based PDFs in one workspace.
- Extract text while preserving original page numbers and source filenames.
- Split each page into overlapping word chunks without crossing page boundaries.
- Create embeddings using `all-MiniLM-L6-v2` and store them in ChromaDB.
- Retrieve semantically relevant evidence across the indexed files.
- Generate grounded answers with Groq, prompting the model to cite `[filename, p. N]`.
- Inspect extracted text and a rendered page preview.
- Generate a first-pass summary or compare documents using retrieved passages.
- Use the left navigation and a collapsible settings area to reduce clutter.
- Includes runtime dependencies and unit tests.

## Requirements

- Python 3.11 or 3.12 recommended.
- Internet access for package installation and the initial download of the embedding model.
- A Groq API key for answer generation. PDF extraction and indexing can run without it, but generated answers require it.

## Install on macOS/Linux

```bash
# Extract the ZIP, then enter the project folder
cd Atlas_Research_Assistant

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Set your own API key in the current terminal session
export GROQ_API_KEY="paste-your-real-groq-key-here"

streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:GROQ_API_KEY="paste-your-real-groq-key-here"
streamlit run app.py
```

## Streamlit Community Cloud

In **Manage app → Settings → Secrets**, add:

```toml
GROQ_API_KEY = "paste-your-real-groq-key-here"
```

Never commit your actual API key. The provided `.gitignore` excludes local secrets files.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The current unit tests verify PDF page-number preservation, multiple-document metadata, and page-aware chunking. They do not call the live Groq API or verify a cloud deployment.

## How to use the dashboard

1. Upload one or more text-based PDFs in the left sidebar.
2. Open **Index settings** if you'd like to change chunk size, overlap, retrieval count, or the Groq model.
3. On **Overview**, click **Build research index**.
4. Open **Research chat** to ask questions, or use **Research tools** for a summary or comparison.
5. Open **Document library** to review the extracted text and page preview.
6. Check the **Sources and evidence** panel to verify an answer against the original PDF page.

If you change chunk size, chunk overlap, or the uploaded files, rebuild the index before continuing.

## Project structure

```text
Atlas_Research_Assistant/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── utils/
│   ├── __init__.py
│   └── pdf_processor.py
└── tests/
    └── test_multi_pdf.py
```

## Architecture

```text
Multiple PDFs
  -> PyMuPDF page-wise extraction
  -> overlapping chunks with filename/page metadata
  -> Sentence Transformers embeddings
  -> ChromaDB cosine-similarity retrieval
  -> Groq LLM answer grounded in retrieved passages
  -> answer + source citations
```

## Known limitations

- Scanned or image-only PDFs need OCR, which is not included yet.
- Retrieval currently uses semantic vector search; hybrid keyword search and reranking are planned future improvements.
- Summary and comparison actions use a bounded set of retrieved passages. They are not guaranteed to inspect every page of every PDF.
- Answers may still be inaccurate; verify important points in the cited original pages.
- The ChromaDB index is local to the running app environment. Some hosting environments may reset local storage, and this prototype is not a multi-user production document store.
- Visitors of a public deployment may consume the quota associated with the configured Groq key; monitor usage and rate limits.
