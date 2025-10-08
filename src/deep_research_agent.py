from langgraph.graph.state import CompiledStateGraph

from src.deep_research_state import DeepResearchState


class DeepResearchAgent:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    def perform_research(
        self, state: DeepResearchState, config: dict = None
    ) -> dict:
        return self.graph.invoke(state, config=config)
