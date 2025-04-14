from dataclasses import dataclass

from adf.agents import Agent


@dataclass(frozen=True)
class Role:
    """Agent role in the system."""

    name: str
    agent: Agent
