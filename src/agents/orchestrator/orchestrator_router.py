from langgraph.constants import END

from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)
from src.agents.topic_extractor_agent import TopicExtractorAgent


class OrchestratorRouter:
    def paths(self, name: str):
        if name == TopicExtractorAgent.name:
            return ["parallel_research", END]
        raise ValueError(f"unknown paths for {name}")

    def route_from(self, name: str):
        if name == TopicExtractorAgent.name:
            return self._route_from_topic_extractor
        raise ValueError(f"unknown paths for {name}")

    def _route_from_topic_extractor(self, state: OrchestratorResearchState):
        if state.unresearched_topics:
            return "parallel_research"
        else:
            return END
