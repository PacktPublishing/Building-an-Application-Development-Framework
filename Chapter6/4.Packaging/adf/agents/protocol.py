from typing import Protocol

from adf.messages import AssistantMessage, Message, SystemMessage


class Agent(Protocol):
    """Generic interface for an agent."""

    name: str
    system_prompt: SystemMessage | None = None

    def generate(self, messages: list[Message]) -> AssistantMessage:
        """Call the agent with a prompt and return the response."""
