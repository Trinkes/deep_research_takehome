from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents.research_agent.research_state import ResearchState
from src.base_agent import BaseAgent
from src.deep_research_state import DeepResearchState


class ResearchAgentResponse(BaseModel):
    to_research: list[str] = Field(
        default_factory=list,
        description="A list of researches that should be executed by the Research Agents",
    )
    answer: list[str] = Field(
        default_factory=list,
        description="a small description of the to_research list to be shown to the user",
    )


class ResearchAgentOrchestrator(BaseAgent):
    name = "research_agent"

    def __init__(self, research_graph: CompiledStateGraph):
        self.research_graph = research_graph

    def __call__(self, state: DeepResearchState):
        research_state = ResearchState(research_topic=state.document)
        result = self.research_graph.invoke(research_state)

        return {"messages": [AIMessage(content=result)]}
