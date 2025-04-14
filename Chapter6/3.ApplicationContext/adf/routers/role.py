from dataclasses import dataclass

from ..agents import Agent


@dataclass(frozen=True)
class Role:
    name: str
    agent: Agent
