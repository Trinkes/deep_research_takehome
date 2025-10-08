from langchain_core.language_models import BaseLanguageModel
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.agents.research_agent.research_agent import ResearchAgent
from src.agents.research_agent.research_state import ResearchState


class ResearchAgentBuilder:
    def __init__(
        self,
    ):
        self._llm = None

    def with_llm(
        self,
        llm: BaseLanguageModel | None = None,
    ) -> "ResearchAgentBuilder":
        self._llm = llm
        return self

    def build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(ResearchState)
        graph_builder.add_node("research", ResearchAgent(llm=self._llm))

        graph_builder.set_entry_point("research")
        graph_builder.set_finish_point("research")

        return graph_builder.compile()
