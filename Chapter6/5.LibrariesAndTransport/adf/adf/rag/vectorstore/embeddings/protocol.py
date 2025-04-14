from collections.abc import Iterable, Sequence
from typing import Protocol


class Embeddings(Protocol):
    """Turns text into one or more dense vectors."""

    dimensions: int

    def embed(self, texts: Iterable[str]) -> Sequence[Sequence[float]]:
        """Turns text into one or more dense vectors."""
