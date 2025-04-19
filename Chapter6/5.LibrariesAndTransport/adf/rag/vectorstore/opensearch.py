import uuid
from collections.abc import Iterable

from opensearchpy import OpenSearch

from adf.rag.vectorstore.protocol import VectorStore


class OpenSearchVectorStore(VectorStore):
    """OpenSearch vector store."""

    _os: OpenSearch
    index_name: str

    def ensure_index(self) -> None:
        if not self._os.indices.exists(self.index_name):
            self._os.indices.create(
                self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "vector": {
                                "type": "knn_vector",
                                "dimension": self.embeddings.dimensions,
                                "method": {"name": "hnsw", "engine": "nmslib"},
                            },
                            "text": {"type": "text"},
                        }
                    }
                },
            )

    def index(self, texts: Iterable[str]) -> None:
        vectors = self.embeddings.embed(texts)
        bulk_body = []
        for text, vec in zip(texts, vectors, strict=False):
            _id = str(uuid.uuid4())
            bulk_body.extend(
                [
                    {"index": {"_index": self.index_name, "_id": _id}},
                    {"vector": vec, "text": text},
                ]
            )
        self._os.bulk(bulk_body, refresh=True)

    def search(self, text: str, k: int = 5) -> list[str]:
        vector = self.embeddings.embed([text])[0]
        resp = self._os.search(
            index=self.index_name,
            body={
                "size": k,
                "query": {"knn": {"vector": {"vector": vector, "k": k}}},
                "_source": ["text"],
            },
        )
        return [hit["_source"]["text"] for hit in resp["hits"]["hits"]]
