from typing import Protocol

from ..messages import AssistantMessage, Message, SystemMessage


class Agent(Protocol):
    system_prompt: SystemMessage | None
    messages: list[Message]

    def generate(self) -> AssistantMessage:
        """Method to call the model with a prompt and return the response."""
