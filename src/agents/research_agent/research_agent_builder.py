from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseLanguageModel
from langchain_tavily import TavilySearch
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Send

from src.agents.research_agent.research_agent import QueryExtractor
from src.agents.research_agent.research_state import ResearchState
from src.agents.search_agent import SearchAgent


class ResearchAgentBuilder:
    def __init__(
        self,
    ):
        self._llm = None
        self._search_tool = None

    def with_llm(
        self,
        llm: BaseLanguageModel | None = None,
    ) -> "ResearchAgentBuilder":
        self._llm = llm
        return self

    def online_search(self, state: ResearchState):
        if len(state.unresearched_queries) == 0:
            return END

        send_operations = []
        for search_query in state.unresearched_queries:
            send_operations.append(Send("online_search", search_query))
        return send_operations

    def with_search_tool(self, search_tool: TavilySearch | DuckDuckGoSearchResults)->"ResearchAgentBuilder":
        self._search_tool = search_tool
        return self

    def build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(ResearchState)
        graph_builder.add_node("query_extractor", QueryExtractor(llm=self._llm))
        graph_builder.add_node("online_search", SearchAgent(search_tool=self._search_tool))

        graph_builder.set_entry_point("query_extractor")
        graph_builder.add_conditional_edges(
            "query_extractor", self.online_search, path_map=["online_search", END]
        )
        graph_builder.add_edge("online_search", "query_extractor")
        graph_builder.set_finish_point("query_extractor")

        return graph_builder.compile()
