from adf.agents import Agent
from adf.messages import SystemMessage, Message, AssistantMessage


class DummyAgent(Agent):
    # Provide the required attributes.
    system_prompt = SystemMessage("dummy prompt")
    name = "dummy_agent"

    def generate(self, messages: list[Message]) -> AssistantMessage:
        # For testing purposes, simply join all message contents with a separator.
        combined = " | ".join(str(msg) for msg in messages)
        return AssistantMessage(content=f"Dummy response: {combined}")