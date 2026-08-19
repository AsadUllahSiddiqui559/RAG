# RAG from scratch — local setup

This directory contains environment scaffolding only. The RAG pipeline is intentionally left for you to implement while following the [CodeAwake tutorial](https://medium.com/@codeawake/rag-from-scratch-ec1a36be0264).

## Requirements

- Python 3.11 (the version used by the tutorial project)
- A Groq API key
- Internet access on the first run so Nomic can download its local embedding model

No LangChain or vector database is needed. The tutorial builds chunking, cosine-similarity search, and the vector store directly.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The environment is already scaffolded in this workspace. Replace the placeholder in `.env` with your Groq API key. Never commit `.env`.

NLTK's Punkt tokenizer data is already installed inside `.venv`. To restore it in a newly-created environment:

```bash
python -m nltk.downloader punkt
```

If newer NLTK code asks for it, also run:

```bash
python -m nltk.downloader punkt_tab
```

To use the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name rag-from-scratch --display-name "Python (RAG from scratch)"
```

## Suggested directories

Use these as your implementation grows:

```text
app/       Python source code (create when you begin implementing)
data/docs/ Source PDF files (ready for your PDFs)
```

## Verify

```bash
python -c "import groq, nltk, nomic, numpy, pdfminer, pydantic_settings, tiktoken, tqdm; print('Environment OK')"
```

The LLM call uses Groq, so generation is not fully offline. Nomic embeddings run locally after the model files have been downloaded once.
