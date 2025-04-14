from ..messages import AssistantMessage, Message
from ..routers import Role


class RoleIsNotRegisteredError(Exception):
    pass


class RoleRouter:
    __registry__: dict[str, Role] = {}

    def register(self, role: Role) -> None:
        """Register a role."""
        self.__registry__[role.name] = role

    def navigate(self, role_name: str, messages: list[Message]) -> AssistantMessage:
        """Route the request to the appropriate model."""
        if role_name not in self.__registry__:
            msg = f"Role {role_name} not recognized."
            raise RoleIsNotRegisteredError(msg)

        role = self.__registry__[role_name]
        role.agent.messages = messages
        return role.agent.generate()
