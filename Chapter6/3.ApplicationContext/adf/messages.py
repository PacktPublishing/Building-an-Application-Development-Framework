import abc
from dataclasses import dataclass

@dataclass(frozen=True)
class Message(abc.ABC):
    """
    Abstract base class for all message types.
    Each message represents a conversational turn.
    """
    content: str



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


