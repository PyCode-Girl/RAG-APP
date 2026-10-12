from src import vector_store
from src.config import settings
from src.schemas import RetrievedChunk
from src.tfidf_index import TfidfIndex

tfidf = TfidfIndex()


def rebuild_tfidf() -> None:
    """Re-read all chunks from ChromaDB and rebuild the TF-IDF index."""
    tfidf.build(vector_store.get_all_documents())


def _normalize(scores: dict) -> dict:
    """Rescale scores to the 0-1 range so both methods can be compared fairly."""
    if not scores:
        return {}
    high, low = max(scores.values()), min(scores.values())
    span = (high - low) or 1.0
    return {key: (value - low) / span for key, value in scores.items()}


def hybrid_search(query: str, k: int, alpha: float | None = None) -> list[RetrievedChunk]:
    alpha = settings.alpha if alpha is None else alpha
    pool = k * 3  # collect extra candidates from each method

    docs, dense_scores, sparse_scores = {}, {}, {}

    for doc, score in vector_store.dense_search(query, pool):
        key = (doc.metadata.get("source"), doc.page_content)
        docs[key] = doc
        dense_scores[key] = score

    for doc, score in tfidf.search(query, pool):
        key = (doc.metadata.get("source"), doc.page_content)
        docs[key] = doc
        sparse_scores[key] = score

    dense_n = _normalize(dense_scores)
    sparse_n = _normalize(sparse_scores)

    combined = {
        key: alpha * dense_n.get(key, 0.0) + (1 - alpha) * sparse_n.get(key, 0.0)
        for key in docs
    }
    best = sorted(combined, key=combined.get, reverse=True)[:k]

    return [
        RetrievedChunk(
            text=docs[key].page_content,
            source=str(docs[key].metadata.get("source", "unknown")),
            score=round(combined[key], 4),
        )
        for key in best
    ]