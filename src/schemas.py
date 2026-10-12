from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float = Field(ge=0)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


class RAGResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]


class IngestResponse(BaseModel):
    files_processed: list[str]
    chunks_added: int