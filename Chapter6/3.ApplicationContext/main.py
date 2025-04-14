import os
from typing import Any

from openai import OpenAI

from adf.agents import Agent
from adf.app import Application
from adf.messages import SystemMessage, UserMessage, Message, AssistantMessage

app = Application(name="poetic")
app.settings.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


class GPT4oMiniAgent(Agent):
    client: OpenAI
    system_prompt: SystemMessage

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

    def generate(self) -> AssistantMessage:
        """
        Prepends the default system prompt, converts all messages
        to OpenAI schema, calls the model, and returns an AssistantMessage.
        """
        openai_messages = [self._convert_to_openai_message(self.system_prompt)]
        for msg in self.messages:
            if openai_message := self._convert_to_openai_message(msg):
                openai_messages.append(openai_message)

        # Send to OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            temperature=0.7,
        )
        return AssistantMessage(content=response.choices[0].message.content or "")


@app.register(role="poet")
class PoetGPT4oMiniAgent(GPT4oMiniAgent):
    client = OpenAI(api_key=app.settings.openai_api_key)
    system_prompt = SystemMessage("You are a poet")


@app.register(role="critic")
class CriticGPT4oMiniAgent(GPT4oMiniAgent):
    client = OpenAI(api_key=app.settings.openai_api_key)
    system_prompt = SystemMessage("You are a critic of poetry")


def main():
    # Step 1: build conversation
    conversation = [UserMessage("Create a haiku")]

    # Step 2: direct to "poet" agent
    answer = app.process("poet", conversation)
    print("Poet answer:", answer)

    # Step 3: user then asks for a critique
    conversation.append(answer)
    conversation.append(UserMessage("Can you critique the haiku?"))

    # Step 4: direct to "critic" agent
    response = app.process("critic", conversation)
    print("Critic answer:", response)
    # Step 5: Improve the haiku
    conversation.append(response)
    conversation.append(UserMessage("Can you improve the haiku?"))
    # Step 6: direct to "poet" agent again
    response = app.process("poet", conversation)
    print("Improved haiku:", response)


if __name__ == "__main__":
    main()
