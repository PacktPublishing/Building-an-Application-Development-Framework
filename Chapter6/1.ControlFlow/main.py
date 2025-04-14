#!/usr/bin/env python3
"""
Step 1: Define basic application abstractions and agent control flow.
"""

import abc
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# ======================
# Core Entities: Message Types
# ======================


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



# ======================
# Control Flow: The Agent
# ======================


class OpenAIAgent:
    system_prompt: SystemMessage = SystemMessage("You are a poet")
    client: OpenAI

    """
    Encapsulates the control flow for communicating with the OpenAI model.
    Takes a sequence of messages and returns an assistant reply.
    """

    @staticmethod
    def _convert_to_openai_message(msg: Message) -> Any:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        elif isinstance(msg, AssistantMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content}
        else:
            raise ValueError(f"Unsupported message: {type(msg)}")

    def generate(self, messages: list[Message]) -> AssistantMessage:
        """
        Accepts a list of Message objects, formats them into OpenAI's schema,
        calls the model, and returns an AssistantMessage.
        """
        openai_messages = [self._convert_to_openai_message(self.system_prompt)]
        for msg in messages:
            if openai_message := self._convert_to_openai_message(msg):
                openai_messages.append(openai_message)

        # Send to OpenAI model
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            temperature=0.7,
        )

        # Return the model’s reply as an AssistantMessage
        return AssistantMessage(content=response.choices[0].message.content or "")


# ======================
# Application Entry Point
# ======================


def main():
    # 1. Create user message (prompt)
    user_message = UserMessage("Create a haiku")

    # 2. Initialize OpenAI API client
    openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # 3. Initialize the agent
    agent = OpenAIAgent()
    agent.client = openai

    # 5. Define the conversation so far
    conversation = [user_message]

    # 6. Call the agent to generate a response
    answer = agent.generate(conversation)

    # 7. Display input and output
    print(f"User input:   {user_message}")
    print("\nAgent answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
