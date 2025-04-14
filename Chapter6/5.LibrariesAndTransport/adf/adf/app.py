from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adf.agents import Agent
from adf.messages import AssistantMessage, Message, SystemMessage, UserMessage
from adf.rag.mixin import RagMixin
from adf.routers import Role, RoleRouter

T = TypeVar("T", bound=Agent)


class _MessageIn(BaseModel):
    role: str
    content: str

    def to_domain(self) -> Message:
        if self.role == "user":
            return UserMessage(self.content)
        if self.role == "system":
            return SystemMessage(self.content)
        if self.role == "assistant":
            return AssistantMessage(self.content)
        msg = f"unknown role {self.role}"
        raise ValueError(msg)


class _AnswerOut(BaseModel):
    content: str


class Document(BaseModel):
    """Represents a document to be indexed in the vector store."""

    content: str


class Application:
    """
    An Application holds:
    - a name (e.g. "poetic")
    - a RoleRouter for dispatching conversations to agents
    - a decorator-based registration mechanism to easily register new roles/agents.
    """

    settings = SimpleNamespace()
    fastapi: FastAPI

    def __init__(self, name: str) -> None:
        """Initialize the Application with a name."""
        self.name = name
        self.router = RoleRouter()
        self.fastapi = FastAPI(title=f"{name}-service")

        @self.fastapi.get("/roles", tags=["agents"])
        async def get_roles() -> list[str]:
            """List all registered roles."""
            return list(self.router.__registry__.keys())  # type: ignore[attr-defined]

    def register(self, role: str) -> Callable[[type[T]], type[T]]:
        """
        Register a new role in the application
        Usage:

            @app.register("poet")
            class PoetAgent(MyAgent):
                system_prompt = SystemMessage("You are a poet")
        """

        def decorator(agent_cls: type[T]) -> type[T]:
            instance = agent_cls()
            self.router.register(Role(name=role, agent=instance))

            base_path = f"/{role}"

            @self.fastapi.post(base_path, response_model=_AnswerOut, tags=["agents"])
            async def _dialog(messages: list[_MessageIn]) -> _AnswerOut:
                try:
                    answer = self.process(role, [m.to_domain() for m in messages])
                except Exception as exc:  # pragma: no cover
                    raise HTTPException(500, str(exc)) from exc
                return _AnswerOut(content=str(answer))

            if isinstance(instance, RagMixin):

                @self.fastapi.post(f"{base_path}/index", response_model=_AnswerOut, tags=["agents"])
                async def _index(documents: list[Document]) -> _AnswerOut:
                    try:
                        instance.index_documents(documents)
                    except Exception as exc:  # pragma: no cover
                        raise HTTPException(500, str(exc)) from exc
                    return _AnswerOut(content="Documents indexed successfully.")

            return agent_cls

        return decorator

    def process(self, role: str, conversation: list[Message]) -> AssistantMessage:
        """Forward the conversation to the router and return the AssistantMessage result."""
        return self.router.navigate(role, conversation)
