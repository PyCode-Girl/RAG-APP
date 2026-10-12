from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src import vector_store
from src.config import settings
from src.generator import generate_answer
from src.ingest import SUPPORTED, chunk_documents, load_file
from src.retriever import hybrid_search, rebuild_tfidf
from src.schemas import IngestResponse, QueryRequest, RAGResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the app starts
    Path(settings.data_dir).mkdir(exist_ok=True)
    rebuild_tfidf()
    yield


app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    processed, total_chunks = [], 0
    for upload in files:
        name = Path(upload.filename).name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {name}. Allowed: {sorted(SUPPORTED)}",
            )
        path = Path(settings.data_dir) / name
        path.write_bytes(await upload.read())
        chunks = chunk_documents(load_file(path))
        vector_store.add_documents(chunks)
        processed.append(name)
        total_chunks += len(chunks)
    rebuild_tfidf()
    return IngestResponse(files_processed=processed, chunks_added=total_chunks)


@app.post("/query", response_model=RAGResponse)
def query(req: QueryRequest):
    chunks = hybrid_search(req.question, req.top_k)
    answer = generate_answer(req.question, chunks)
    return RAGResponse(answer=answer, sources=chunks)