from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

store = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory=settings.chroma_dir,
    collection_metadata={"hnsw:space": "cosine"},
)


def add_documents(docs: list[Document]) -> None:
    if docs:
        store.add_documents(docs)


def dense_search(query: str, k: int) -> list[tuple[Document, float]]:
    results = store.similarity_search_with_score(query, k=k)
    # Chroma returns a distance (smaller = closer); convert to similarity (bigger = closer)
    return [(doc, max(0.0, 1.0 - dist)) for doc, dist in results]


def get_all_documents() -> list[Document]:
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]