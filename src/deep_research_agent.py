from langgraph.graph.state import CompiledStateGraph

from src.deep_research_state import DeepResearchState


class DeepResearchAgent:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    async def perform_research(self, state: DeepResearchState):
        return await self.graph.ainvoke(state)
