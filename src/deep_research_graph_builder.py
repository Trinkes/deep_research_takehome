from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from typing_extensions import TypeAlias

from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)
from src.agents.scoping_agent import ScopingAgent
from src.deep_research_state import DeepResearchState

LanggraphNode: TypeAlias = Callable[[DeepResearchState], dict | DeepResearchState]


class DeepResearchGraphBuilder:
    def __init__(
        self,
    ):
        self._llm = None
        self._orchestrator: CompiledStateGraph | None = None

    def with_orchestrator(
        self, orchestrator: CompiledStateGraph | None
    ) -> "DeepResearchGraphBuilder":
        self._orchestrator = orchestrator
        return self

    def build_graph(self) -> CompiledStateGraph:
        if self._llm is None:
            raise ValueError(
                "LLM must be configured using with_llm() before building graph"
            )
        if self._orchestrator is None:
            raise ValueError(
                "Orchestrator must be configured using with_orchestrator() before building graph"
            )

        graph_builder = StateGraph(DeepResearchState)
        graph_builder.add_node("scoping_agent", ScopingAgent(self._llm))
        graph_builder.add_node(
            "research_agent_orchestrator", self.research_agent_orchestrator
        )

        graph_builder.set_entry_point("scoping_agent")
        graph_builder.add_conditional_edges(
            "scoping_agent",
            self._route_from_scoping,
            ["research_agent_orchestrator", "scoping_agent"],
        )
        graph_builder.set_finish_point("research_agent_orchestrator")

        return graph_builder.compile(checkpointer=MemorySaver())

    def research_agent_orchestrator(self, state: DeepResearchState, config: RunnableConfig | None = None) -> dict:
        research_state = OrchestratorResearchState(
            research_description=state.document,
            max_generated_topics=state.max_generated_topics,
            max_queries_per_topic=state.max_queries_per_topic,
        )
        results = self._orchestrator.invoke(research_state, config=config)
        return {"messages": [AIMessage(content=results["research_report"])]}

    def _route_from_scoping(self, state: DeepResearchState) -> str:
        if state.needs_research_clarification or state.document is None:
            return "scoping_agent"
        else:
            return "research_agent_orchestrator"

    def with_llm(
        self,
        llm: BaseChatModel | None = None,
    ) -> "DeepResearchGraphBuilder":
        self._llm = llm
        return self
