from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class DeepResearchState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list, description="Chat history"
    )
    document: str | None = Field(
        default=None,
        description="the full context document that will be use further to guide the research",
    )
    needs_research_clarification: bool = Field(
        default=False,
        description="whether or not the user needs to provide additional clarification about the research scope or context",
    )
    max_generated_topics: int = Field(
        default=4, description="max generated topics per research"
    )
    max_queries_per_topic: int = Field(
        default=2,
        description="max queries to be performed per generated topic in the research",
    )
