from transformers import pipeline

from src.config import settings
from src.schemas import RetrievedChunk

_pipe = None


def _get_pipe():
    """Load the language model once, the first time it is needed."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline("text-generation", model=settings.llm_model)
    return _pipe


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I couldn't find anything relevant in the documents."

    context = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
    messages = [
        {
            "role": "system",
            "content": "Answer only from the context provided. "
            "If the answer is not in the context, say you don't know.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    output = _get_pipe()(messages, max_new_tokens=300, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()