from typing import Protocol

from ..messages import AssistantMessage, Message, SystemMessage


class Agent(Protocol):
    name: str
    system_prompt: SystemMessage | None = None

    def generate(self, messages: list[Message]) -> AssistantMessage:
        """Method to call the model with a prompt and return the response."""
