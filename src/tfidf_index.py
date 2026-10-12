import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        self.docs: list[Document] = []

    def build(self, docs: list[Document]) -> None:
        self.docs = docs
        if docs:
            self.matrix = self.vectorizer.fit_transform(
                [d.page_content for d in docs]
            )
        else:
            self.matrix = None

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in top]