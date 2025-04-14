import abc
from dataclasses import dataclass


class Message(abc.ABC):
    """
    Abstract base class for all message types.
    Each message represents a conversational turn.
    """
    content: str

    def __str__(self):
        return self.content


@dataclass(frozen=True)
class UserMessage(Message):
    """
    Represents a message coming from the user.
    """


@dataclass(frozen=True)
class SystemMessage(Message):
    """
    Represents the system prompt that sets up model behavior.
    """


@dataclass(frozen=True)
class AssistantMessage(Message):
    """
    Represents a message coming from the assistant (model).
    """

