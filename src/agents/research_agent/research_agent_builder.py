from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseLanguageModel
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

    def build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(ResearchState)
        # search = TavilySearch(max_results=1, include_raw_content=True, )
        search = DuckDuckGoSearchResults(output_format="list")
        graph_builder.add_node("query_extractor", QueryExtractor(llm=self._llm))
        graph_builder.add_node("online_search", SearchAgent(search_tool=search))

        graph_builder.set_entry_point("query_extractor")
        graph_builder.add_conditional_edges(
            "query_extractor", self.online_search, path_map=[]
        )
        graph_builder.set_finish_point("online_search")

        return graph_builder.compile()
