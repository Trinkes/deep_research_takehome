from operator import add
from typing import Annotated

from pydantic import BaseModel, Field

from src.agents.research_agent.research_state import ResearchState


class OrchestratorResearchState(BaseModel):
    research_description: str = Field(
        description="the full context document to use as guide for the research"
    )
    searched_topics: Annotated[list[str], add] = Field(
        default_factory=list, description="list of all discovered topics"
    )
    results: Annotated[list[ResearchState], add] = Field(
        default_factory=list, description="list of research results"
    )
    research_report: str | None = Field(
        default=None, description="The completed research result"
    )
    max_queries_per_topic: int = Field(
        default=2,
        description="max queries to be performed per generated topic in the research",
    )
    max_generated_topics: int = Field(
        default=4, description="max generated topics per research"
    )

    @property
    def unresearched_topics(self) -> list[str]:
        researched = {result.research_topic for result in self.results}
        return [topic for topic in self.searched_topics if topic not in researched]
