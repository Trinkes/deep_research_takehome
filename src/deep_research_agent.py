from langgraph.graph.state import CompiledStateGraph

from src.deep_research_state import DeepResearchState

DEFAULT_RECURSION_LIMIT = 25


class DeepResearchAgent:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    def perform_research(
        self, state: DeepResearchState, config: dict | None = None
    ) -> dict:
        if config is None:
            config = {}

        if "recursion_limit" not in config:
            config["recursion_limit"] = DEFAULT_RECURSION_LIMIT

        return self.graph.invoke(state, config=config)
