import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

from adf.agents import Agent
from adf.app import Application
from adf.messages import AssistantMessage, Message, SystemMessage, UserMessage
from adf.rag.mixin import RagMixin
from adf.rag.vectorstore.opensearch import OpenSearchVectorStore


load_dotenv()

app = Application(name="poetic")
app.settings.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
app.settings.opensearch_host = os.environ.get("OPENSEARCH_HOST", "")
app.settings.opensearch_port = int(os.environ.get("OPENSEARCH_PORT", "9200"))


opensearch = OpenSearch(
    hosts=[
        {"host": app.settings.opensearch_host, "port": app.settings.opensearch_port}
    ],
)


class GPT4oMiniAgent(Agent, RagMixin):
    system_prompt: SystemMessage

    @staticmethod
    def _convert_to_openai_message(msg: Message) -> Any:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if isinstance(msg, AssistantMessage):
            return {"role": "assistant", "content": msg.content}
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content}
        msg = f"Unsupported message type: {type(msg)}"
        raise ValueError(msg)

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


@app.register(role="critic")
class CriticGPT4oMiniAgent(GPT4oMiniAgent):
    system_prompt = SystemMessage("You are a critic of poetry")
    vectorstore = OpenSearchVectorStore(opensearch, "critic")
    client = OpenAI(api_key=app.settings.openai_api_key)


@app.register(role="poet")
class PoetGPT4oMiniAgent(GPT4oMiniAgent):
    system_prompt = SystemMessage("You are a poet")
    vectorstore = OpenSearchVectorStore(opensearch, "poetic")
    client = OpenAI(api_key=app.settings.openai_api_key)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app.fastapi, host="0.0.0.0", port=8000)
