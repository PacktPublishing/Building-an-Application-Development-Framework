from typing import ClassVar

from adf.messages import AssistantMessage, Message
from adf.routers import Role


class RoleIsNotRegisteredError(Exception):
    """Exception raised when a role is not registered in the router."""


class RoleRouter:
    """Router to manage roles and route requests to the appropriate agent."""

    __registry__: ClassVar[dict[str, Role]] = {}

    def register(self, role: Role) -> None:
        """Register a role."""
        self.__registry__[role.name] = role

    def navigate(self, role_name: str, messages: list[Message]) -> AssistantMessage:
        """Route the request to the appropriate model."""
        if role_name not in self.__registry__:
            msg = f"Role {role_name} not recognized."
            raise RoleIsNotRegisteredError(msg)

        role = self.__registry__[role_name]
        return role.agent.generate(messages)
