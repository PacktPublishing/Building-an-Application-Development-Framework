from types import SimpleNamespace

from .messages import Message, AssistantMessage
from adf.routers import Role, RoleRouter
from adf.vectorstore import SimpleVectorStore, VectorStore


class Application:
    """
    An Application holds:
    - a name (e.g. "poetic")
    - a RoleRouter for dispatching conversations to agents
    - a decorator-based registration mechanism to easily register new roles/agents.
    - a vector store for document retrieval and similarity search
    """

    settings = SimpleNamespace()

    def __init__(self, name: str, vector_store: VectorStore = None) -> None:
        self.name = name
        self.router = RoleRouter()
        self.vector_store = vector_store or SimpleVectorStore()

    def register(self, role: str):
        """
        A decorator that accepts a role name (e.g. "poet"),
        instantiates the decorated Agent class, and registers it in the RoleRouter.

        Usage:

            @app.register(role="poet")
            class PoetGPT4oMiniAgent(GPT4oMiniAgent):
                system_prompt = SystemMessage("You are a poet")

        """

        def decorator(agent_cls):
            # Instantiate the agent
            instance = agent_cls()

            # If needed, we can check or set a system prompt here:
            # if not instance.is_ready:
            #     instance.instruct(SystemMessage("You are ..."))

            # Register this instance with the router
            self.router.register(Role(name=role, agent=instance))
            return agent_cls

        return decorator

    def process(self, role: str, conversation: list[Message]) -> AssistantMessage:
        """
        A simple convenience method to forward the conversation
        to the router and return the AssistantMessage result.
        """
        return self.router.navigate(role, conversation)

    def index_documents(self, documents: list[str]) -> None:
        """
        Add documents to the vector store.
        """
        self.vector_store.index(documents)

    def search_documents(self, query: str, k: int = 5) -> list[str]:
        """
        Search for similar documents in the vector store.
        """
        return self.vector_store.search(query, k)
