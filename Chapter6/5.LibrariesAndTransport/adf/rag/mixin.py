from adf.messages import Message, SystemMessage, UserMessage
from adf.rag.vectorstore.protocol import VectorStore


class RagMixin:
    """RAG mixin for agents."""

    vectorstore: VectorStore

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        docs = self.vectorstore.search(query, k=k)
        return "\n".join(docs)

    def _augment(self, messages: list[Message]) -> list[Message]:
        query = next(m.content for m in reversed(messages) if isinstance(m, UserMessage))
        ctx = self._retrieve_context(query)
        ctx_msg = SystemMessage(f"Context:\n{ctx}")
        return [ctx_msg, *messages]
