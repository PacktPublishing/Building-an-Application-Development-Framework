from adf.messages import Message, SystemMessage, UserMessage
from adf.rag.vectorstore.protocol import VectorStore


class RagMixin:
    """RAG mixin for agents."""

    vectorstore: VectorStore
    _messages: list[Message]

    def index_documents(self, texts: list[str]) -> None:
        self.vectorstore.ensure_index()
        self.vectorstore.index(texts)

    def retrieve_context(self, query: str, k: int = 5) -> str:
        docs = self.vectorstore.search(query, k=k)
        return "\n".join(docs)

    @property
    def messages(self) -> list[Message]:
        query = next(m.content for m in reversed(self._messages) if isinstance(m, UserMessage))
        ctx = self.retrieve_context(query)
        ctx_msg = SystemMessage(f"Context:\n{ctx}")
        return [ctx_msg, *self._messages]
