# Hybrid RAG App

A question-answering API that searches your own documents and answers using only what it finds.

## How it works

1. **Ingest:** upload PDF, Word, text, Markdown, CSV or HTML files, which are split into chunks.
2. **Index:** every chunk goes into two search indexes:
   - TF-IDF (scikit-learn) for exact keyword matches
   - Hugging Face embeddings stored in ChromaDB for matches by meaning
3. **Retrieve:** a hybrid search mixes both scores.
4. **Generate:** a small Hugging Face language model (Qwen2.5-0.5B-Instruct) writes an answer from the retrieved chunks and shows its sources.

## Tech stack

Python, FastAPI, LangChain, ChromaDB, Hugging Face Transformers, scikit-learn, Pydantic

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Check that the server is running |
| POST | `/ingest` | Upload one or more documents |
| POST | `/query` | Ask a question and get an answer with sources |