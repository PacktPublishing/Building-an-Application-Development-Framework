from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeVar

from adf.agents import Agent
from adf.messages import AssistantMessage, Message
from adf.routers import Role, RoleRouter

T = TypeVar("T", bound=Agent)


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
            return agent_cls

        return decorator

    def process(self, role: str, conversation: list[Message]) -> AssistantMessage:
        """
        A simple convenience method to forward the conversation
        to the router and return the AssistantMessage result.
        """
        return self.router.navigate(role, conversation)
