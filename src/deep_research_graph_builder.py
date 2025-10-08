from typing import Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from typing_extensions import TypeAlias

from src.agents.orchestrator.orchestrator_research_agent_graph_builder import (
    ResearchAgentOrchestratorGraphBuilder,
)
from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)
from src.agents.research_agent.research_agent_builder import ResearchAgentBuilder
from src.agents.scoping_agent import ScopingAgent
from src.deep_research_state import DeepResearchState

LanggraphNode: TypeAlias = Callable[[DeepResearchState], dict | DeepResearchState]


class DeepResearchGraphBuilder:
    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
    ):
        self.llm = llm

        self._research_agent: CompiledStateGraph = None
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
            research_graph = ResearchAgentBuilder().with_llm(llm=self.llm).build_graph()
            self._research_agent = (
                ResearchAgentOrchestratorGraphBuilder()
                .with_llm(self.llm)
                .with_research_graph(research_graph)
                .build_graph()
            )

        graph_builder.add_node(
            "research_agent_orchestrator", self.research_agent_orchestrator
        )

        graph_builder.set_entry_point(ScopingAgent.name)
        graph_builder.add_conditional_edges(
            ScopingAgent.name,
            self._route_from_scoping,
            ["research_agent_orchestrator", ScopingAgent.name],
        )
        graph_builder.set_finish_point("research_agent_orchestrator")

        return graph_builder.compile(checkpointer=MemorySaver())

    def research_agent_orchestrator(self, state: DeepResearchState):
        research_state = OrchestratorResearchState(
            research_description=state.document,
            max_generated_topics=state.max_generated_topics,
            max_queries_per_topic=state.max_queries_per_topic,
        )
        results = self._research_agent.invoke(research_state)
        return {"messages": [AIMessage(content=results["research_report"])]}

    def _route_from_scoping(self, state: DeepResearchState) -> str:
        if state.needs_research_clarification:
            return ScopingAgent.name
        else:
            return "research_agent_orchestrator"
