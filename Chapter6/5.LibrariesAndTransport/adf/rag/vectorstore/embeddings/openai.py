from collections.abc import Iterable, Sequence

from openai import OpenAI

from adf.rag.vectorstore.embeddings.protocol import Embeddings


class OpenAIEmbeddings(Embeddings):
    """OpenAI embedding model."""

    dimensions = 768
    model: str = "text-embedding-3-small"
    openai_client: OpenAI
    batch: int = 100

    def embed(self, texts: Iterable[str]) -> Sequence[Sequence[float]]:
        out = []
        for chunk in self._chunk(texts, self.batch):
            out.extend(self._call(chunk))
        return out

    @staticmethod
    def _chunk(it: Iterable[str], n: int) -> Iterable[list[str]]:
        chunk = []
        for item in it:
            chunk.append(item)
            if len(chunk) == n:
                yield chunk
                chunk.clear()
        if chunk:
            yield chunk

    def _call(self, batch: list[str]) -> list[list[float]]:
        resp = self.openai_client.embeddings.create(model=self.model, input=batch, dimensions=self.dimensions)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
