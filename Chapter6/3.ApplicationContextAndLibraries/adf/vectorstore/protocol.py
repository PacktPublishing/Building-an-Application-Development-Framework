from typing import Protocol


class VectorStore(Protocol):
    def index(self, documents: list[str]) -> None:
        """Add documents to the vector store."""
        pass

    def search(self, query: str, k: int = 5) -> list[str]:
        """Search for similar documents in the vector store."""
        pass
