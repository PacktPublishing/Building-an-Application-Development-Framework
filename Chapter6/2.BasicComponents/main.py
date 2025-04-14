#!/usr/bin/env python3
"""
Step 2: Basic role-based routing for an agent system (revised).
"""

import abc
import os
from dataclasses import dataclass
from typing import Protocol, Any

from openai import OpenAI


### Core Entities ###

@dataclass(frozen=True)
class Message(abc.ABC):
    """Abstract base class for all message types."""

    content: str


@dataclass(frozen=True)
class UserMessage(Message):
    """Represents input from a user."""


@dataclass(frozen=True)
class SystemMessage(Message):
    """Represents system instructions/context for the agent."""



@dataclass(frozen=True)
class AssistantMessage(Message):
    """Represents output from the agent."""



### Abstract Components ###


class AgentIsNotInstructedError(Exception):
    """Raised if the agent does not receive a proper system message first."""

    pass


class Agent(Protocol):
    """Generic interface for an agent."""
    system_prompt: SystemMessage
    messages: list[Message]

    def generate(self) -> AssistantMessage: ...


### Control Flow: Role Router ###


class RoleIsNotRegisteredError(Exception):
    """Raised if attempting to route to an unknown role."""

    pass


@dataclass(frozen=True)
class Role:
    """Links a role name to a specific agent."""

    name: str
    agent: Agent


class RoleRouter:
    """Routes messages to the correct agent based on the requested role."""

    __registry__: dict[str, Role] = {}

    def register(self, role: Role) -> None:
        """Register a new role in the router."""
        self.__registry__[role.name] = role

    def navigate(self, role: str, messages: list[Message]) -> AssistantMessage:
        """Look up the agent for `role` and delegate the messages."""
        if role not in self.__registry__:
            raise RoleIsNotRegisteredError(f"Role '{role}' not recognized.")

        target_role = self.__registry__[role]
        return target_role.agent.generate(messages)


### GPT4oMiniAgent Implementation ###


class GPT4oMiniAgent(Agent):
    system_prompt: SystemMessage = SystemMessage("You are a poet")
    client: OpenAI
    """
    An Agent that calls the 'gpt-4o-mini' model.
    It automatically inserts its default system prompt as the first message.
    """

    @staticmethod
    def _convert_to_openai_message(msg: Message) -> Any:
        match msg:
            case SystemMessage():
                return {"role": "system", "content": msg.content}
            case AssistantMessage():
                return {"role": "assistant", "content": msg.content}
            case UserMessage():
                return {"role": "user", "content": msg.content}
            case _:
                raise ValueError(f"Unsupported message type: {type(msg)}")

    def generate(self, messages: list[Message]) -> AssistantMessage:
        """
        Prepends the default system prompt, converts all messages
        to OpenAI schema, calls the model, and returns an AssistantMessage.
        """
        openai_messages = [self._convert_to_openai_message(self.system_prompt)]
        for msg in messages:
            if openai_message := self._convert_to_openai_message(msg):
                openai_messages.append(openai_message)

        # Send to OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            temperature=0.7,
        )

        # Return the model’s reply as an AssistantMessage
        return AssistantMessage(content=response.choices[0].message.content or "")


### Main Entry Point ###


def main():
    # Create an agent and define its role
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai = OpenAI(api_key=openai_api_key)

    agent = GPT4oMiniAgent()
    agent.client = openai

    poet_role = Role(name="poet", agent=agent)

    # Register the role
    router = RoleRouter()
    router.register(poet_role)

    # Build a conversation (no user system prompt needed; agent adds its own)
    user_message = UserMessage("Create a haiku")
    conversation = [user_message]

    # Route the conversation
    answer = router.navigate("poet", conversation)

    print(f"User input: {user_message}")
    print("\nAgent answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
