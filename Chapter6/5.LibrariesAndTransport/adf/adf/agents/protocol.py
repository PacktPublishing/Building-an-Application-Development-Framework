from typing import Protocol

from adf.messages import AssistantMessage, Message, SystemMessage


class Agent(Protocol):
    """Generic interface for an agent."""

    system_prompt: SystemMessage
    _messages: list[Message]

    @property
    def messages(self) -> list[Message]:
        """List of messages exchanged with the agent."""
        return self._messages

    @messages.setter
    def messages(self, messages: list[Message]) -> None:
        """Set the list of messages exchanged with the agent."""
        self._messages = messages

    def generate(self) -> AssistantMessage:
        """Method to call the model with a prompt and return the response."""
