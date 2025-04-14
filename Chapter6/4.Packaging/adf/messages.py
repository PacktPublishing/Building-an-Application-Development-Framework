from dataclasses import dataclass


class Message:
    """Represents a message in the conversation."""


@dataclass(frozen=True)
class UserMessage(Message):
    """Represents the user's text input."""

    content: str

    def __str__(self) -> str:
        return self.content


@dataclass(frozen=True)
class SystemMessage(Message):
    """Represents the 'system' role instructions given to the model."""

    content: str

    def __str__(self) -> str:
        return self.content


@dataclass(frozen=True)
class AssistantMessage(Message):
    """Represents the assistant/model's reply."""

    content: str

    def __str__(self) -> str:
        return self.content
