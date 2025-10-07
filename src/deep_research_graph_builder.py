from typing import Callable

from langchain_core.language_models import BaseLanguageModel
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from typing_extensions import TypeAlias

from src.agents.scoping_agent import ScopingAgent
from src.deep_research_state import DeepResearchState

LanggraphNode: TypeAlias = Callable[[DeepResearchState], dict | DeepResearchState]


class DeepResearchGraphBuilder:
    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm = llm

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
        graph_builder.add_node("scoping_agent", self._scoping_agent)

        graph_builder.add_edge(START, "scoping_agent")
        graph_builder.add_edge("scoping_agent", END)

        return graph_builder.compile()
