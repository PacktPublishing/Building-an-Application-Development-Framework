from typing import Protocol

from adf.messages import AssistantMessage, Message, SystemMessage


class Agent(Protocol):
    """Generic interface for an agent."""

    name: str
    system_prompt: SystemMessage

    def generate(self, messages: list[Message]) -> AssistantMessage:
        """Method to call the model with a prompt and return the response."""
