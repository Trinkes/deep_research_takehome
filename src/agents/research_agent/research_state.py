from operator import add
from typing import Sequence, Annotated

from pydantic import BaseModel, Field
from pydantic_core import Url


class ResearchResult(BaseModel):
    url: Url | None = None
    query: str
    title: str
    content: str | None = None


class ResearchState(BaseModel):
    research_topic: str = Field(
        description="detailed description of the topic to be researched"
    )
    queries: Annotated[Sequence[str], add] = Field(
        default_factory=list, description="list of search queries already performed"
    )
    query_results: Annotated[Sequence[ResearchResult], add] = Field(
        default_factory=list, description="list of raw results from the queries"
    )
    max_queries: int = Field(
        default=2, description="max amount of queries to be performed on a topic"
    )

    @property
    def unresearched_queries(self) -> list[str]:
        researched = {result.query for result in self.query_results}
        return [topic for topic in self.queries if topic not in researched]
