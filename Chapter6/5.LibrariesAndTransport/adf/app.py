from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adf.agents import Agent
from adf.messages import AssistantMessage, Message, SystemMessage, UserMessage
from adf.routers import Role, RoleRouter

T = TypeVar("T", bound=Agent)


class _MessageIn(BaseModel):
    role: str
    content: str

    def to_domain(self) -> Message:
        match self.role:
            case "user":
                return UserMessage(self.content)
            case "system":
                return SystemMessage(self.content)
            case "assistant":
                return AssistantMessage(self.content)
            case _:
                msg = f"unknown role {self.role}"
                raise ValueError(msg)


class _AnswerOut(BaseModel):
    content: str


class Application:
    """
    An Application holds:
    - a name (e.g. "poetic")
    - a RoleRouter for dispatching conversations to agents
    - a decorator-based registration mechanism to easily register new roles/agents.
    """

    settings = SimpleNamespace()

    def __init__(self, name: str) -> None:
        """Initialize the Application with a name."""
        self.name = name
        self.router = RoleRouter()
        self.fastapi = FastAPI(title=f"{name}-service")

    def register(self, role: str) -> Callable[[type[T]], type[T]]:
        """
        A decorator that accepts a role name (e.g. "poet"),
        instantiates the decorated Agent class, and registers it in the RoleRouter.

        Usage:

            @app.register(role="poet")
            class PoetGPT4oMiniAgent(GPT4oMiniAgent):
                system_prompt = SystemMessage("You are a poet")

        """

        def decorator(agent_cls: type[T]) -> type[T]:
            instance = agent_cls()
            self.router.register(Role(name=role, agent=instance))

            path = f"/{role}"

            @self.fastapi.post(path, response_model=_AnswerOut, tags=["agents"])
            async def _endpoint(messages: list[_MessageIn]) -> _AnswerOut:
                try:
                    answer = self.process(role, [m.to_domain() for m in messages])
                except Exception as exc:
                    raise HTTPException(500, str(exc)) from exc
                return _AnswerOut(content=str(answer))

            return agent_cls

        return decorator

    def process(self, role: str, conversation: list[Message]) -> AssistantMessage:
        """
        A simple convenience method to forward the conversation
        to the router and return the AssistantMessage result.
        """
        return self.router.navigate(role, conversation)
