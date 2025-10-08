from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Send

from src.agents.answer_agent import AnswerAgent
from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)
from src.agents.research_agent.research_state import ResearchState
from src.agents.topic_extractor_agent import TopicExtractorAgent


class ResearchAgentOrchestratorGraphBuilder:
    def __init__(
        self,
    ):
        self._research_graph: CompiledStateGraph | None = None
        self._llm: BaseLanguageModel | None = None

    def with_llm(
        self,
        llm: BaseLanguageModel | None = None,
    ) -> "ResearchAgentOrchestratorGraphBuilder":
        self._llm = llm
        return self

    def with_research_graph(
        self, research_graph: CompiledStateGraph
    ) -> "ResearchAgentOrchestratorGraphBuilder":
        self._research_graph = research_graph
        return self

    def build_graph(self) -> CompiledStateGraph:
        if self._llm is None:
            raise ValueError(
                "LLM must be configured using with_llm() before building graph"
            )
        if self._research_graph is None:
            raise ValueError(
                "Research graph must be configured using with_research_graph() before building graph"
            )

        graph_builder = StateGraph(OrchestratorResearchState)

        graph_builder.add_node("topic_extractor_agent", TopicExtractorAgent(self._llm))
        graph_builder.add_node("research", self.research)
        graph_builder.add_node("answer", AnswerAgent(self._llm))

        graph_builder.set_entry_point("topic_extractor_agent")

        graph_builder.add_conditional_edges(
            "topic_extractor_agent", self.parallel_research, ["research", "answer"]
        )
        graph_builder.add_edge(
            "research",
            "topic_extractor_agent",
        )
        graph_builder.set_finish_point("answer")

        return graph_builder.compile()

    def research(self, state: ResearchState, config: RunnableConfig | None = None):
        results = self._research_graph.invoke(state, config=config)
        return {"results": [results]}

    def parallel_research(self, state: OrchestratorResearchState):
        if len(state.unresearched_topics) == 0:
            return "answer"
        send_operations = []
        for topic in state.unresearched_topics:
            send_operations.append(
                Send(
                    "research",
                    ResearchState(
                        research_topic=topic, max_queries=state.max_queries_per_topic
                    ),
                )
            )

        return send_operations
