import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from opensearchpy import OpenSearch

from adf.rag.vectorstore.protocol import VectorStore


@dataclass
class OpenSearchVectorStore(VectorStore):
    """OpenSearch vector store."""

    os: OpenSearch
    index_name: str

    def ensure_index(self) -> None:
        if not self.os.indices.exists(self.index_name):
            self.os.indices.create(
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
        self.os.bulk(bulk_body, refresh=True)

    def search(self, text: str, k: int = 5) -> list[str]:
        vector = self.embeddings.embed([text])[0]
        resp = self.os.search(
            index=self.index_name,
            body={
                "size": k,
                "query": {"knn": {"vector": {"vector": vector, "k": k}}},
                "_source": ["text"],
            },
        )
        return [hit["_source"]["text"] for hit in resp["hits"]["hits"]]
