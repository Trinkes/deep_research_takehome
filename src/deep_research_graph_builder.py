from typing import Callable

from langchain_core.language_models import BaseLanguageModel
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from typing_extensions import TypeAlias

from src.agents.research_agent import ResearchAgent
from src.agents.scoping_agent import ScopingAgent
from src.deep_research_router import DeepResearchRouter
from src.deep_research_state import DeepResearchState

LanggraphNode: TypeAlias = Callable[[DeepResearchState], dict | DeepResearchState]


class DeepResearchGraphBuilder:
    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm = llm

        self._research_agent = None
        self._scoping_agent = None

    def with_scoping_agent(
        self, scoping: LanggraphNode | None
    ) -> "DeepResearchGraphBuilder":
        self._scoping_agent = scoping
        return self

    def build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(DeepResearchState)

        if self._scoping_agent is None:
            self._scoping_agent = ScopingAgent(self.llm)
        graph_builder.add_node(ScopingAgent.name, self._scoping_agent)

        if self._research_agent is None:
            self._research_agent = ResearchAgent()
        graph_builder.add_node(ResearchAgent.name, self._research_agent)

        router = DeepResearchRouter()

        graph_builder.add_edge(START, ScopingAgent.name)
        graph_builder.add_conditional_edges(
            ScopingAgent.name,
            router.route_from(ScopingAgent.name),
            router.paths(ScopingAgent.name),
        )
        graph_builder.add_edge(ResearchAgent.name, END)

        return graph_builder.compile()
