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
