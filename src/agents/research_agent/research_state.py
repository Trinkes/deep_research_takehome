from operator import add
from typing import Sequence, Annotated

from pydantic import BaseModel, Field
from pydantic_core import Url


class ResearchResult(BaseModel):
    url: Url
    query: str
    title: str
    content: str | None = None


class ResearchState(BaseModel):
    research_topic: str = Field(
        description="detailed description of the topic to be researched"
    )
    performed_queries: Annotated[Sequence[str], add] = Field(
        default_factory=list, description="list of search queries already performed"
    )
    query_results: Annotated[Sequence[ResearchResult], add] = Field(
        default_factory=list, description="list of raw results from the queries"
    )
