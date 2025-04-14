from collections.abc import Iterable
from typing import Protocol

from adf.rag.vectorstore.embeddings.protocol import Embeddings


class VectorStore(Protocol):
    """Stores vectors, returns texts of the most similar items."""

    embeddings: Embeddings

    def index(self, texts: Iterable[str]) -> None:
        """Stores vectors, returns texts of the most similar items."""

    def search(self, query: str, k: int = 5) -> list[str]:
        """Returns texts of the most similar items."""
