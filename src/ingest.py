from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": lambda p: Docx2txtLoader(p),
    ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
    ".html": lambda p: BSHTMLLoader(p, open_encoding="utf-8"),
}
SUPPORTED = set(LOADERS)


def load_file(path: Path) -> list[Document]:
    loader = LOADERS[path.suffix.lower()](str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)