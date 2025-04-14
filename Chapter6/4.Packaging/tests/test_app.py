import pytest

from adf.app import Application
from adf.messages import UserMessage, SystemMessage, AssistantMessage
from adf.routers.role import Role
from adf.routers.router import RoleRouter, RoleIsNotRegisteredError
from tests.conftest import DummyAgent


def test_role_router_navigate_success():
    """
    Test that RoleRouter properly routes a conversation when the role is registered.
    """
    # Initialize the router and register a dummy role.
    router = RoleRouter()
    dummy_role = Role(name="test", agent=DummyAgent())
    router.register(dummy_role)

    # Build a test conversation.
    conversation = [SystemMessage("Test system"), UserMessage("Test user")]
    result = router.navigate("test", conversation)

    # Verify that the returned message is an AssistantMessage with the expected content.
    assert isinstance(result, AssistantMessage)
    assert "Dummy response:" in result.content


def test_role_router_navigate_unregistered():
    """
    Test that navigating to an unregistered role raises RoleIsNotRegisteredError.
    """
    router = RoleRouter()
    conversation = [SystemMessage("Test system"), UserMessage("Test user")]

    with pytest.raises(RoleIsNotRegisteredError):
        router.navigate("nonexistent", conversation)


def test_application_registration_and_process():
    """
    Test that the Application's registration decorator registers an agent,
    and the process method correctly routes the conversation.
    """
    # Create an Application instance.
    app = Application(name="test_app")

    # Use the decorator to register a dummy agent under the role "dummy".
    @app.register(role="dummy")
    class DummyAgentForApp(DummyAgent):
        system_prompt = SystemMessage("App dummy prompt")

    # Build a sample conversation.
    conversation = [UserMessage("Hello")]
    result = app.process("dummy", conversation)

    # Verify that the result is an AssistantMessage and that dummy response is present.
    assert isinstance(result, AssistantMessage)
    assert "Dummy response:" in result.content
