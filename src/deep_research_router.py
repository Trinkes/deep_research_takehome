from langgraph.constants import END

from src.agents.research_agent import ResearchAgent
from src.agents.scoping_agent import ScopingAgent
from src.deep_research_state import DeepResearchState


class DeepResearchRouter:
    def _route_from_scoping(self, state: DeepResearchState) -> str:
        if state.needs_research_clarification:
            return END
        else:
            return ResearchAgent.name

    def paths(self, name: str):
        if name == ScopingAgent.name:
            return [ResearchAgent.name, END]
        raise ValueError(f"unknown paths for {name}")

    def route_from(self, name: str):
        if name == ScopingAgent.name:
            return self._route_from_scoping
        raise ValueError(f"unknown paths for {name}")
