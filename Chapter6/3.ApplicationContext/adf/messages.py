
from dataclasses import dataclass


class Message:
    pass


@dataclass(frozen=True)
class UserMessage(Message):
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass(frozen=True)
class SystemMessage(Message):
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass(frozen=True)
class AssistantMessage(Message):
    content: str

    def __str__(self) -> str:
        return self.content
