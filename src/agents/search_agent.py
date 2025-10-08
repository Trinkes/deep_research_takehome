from langchain_core.tools import BaseTool

from src.agents.research_agent.research_state import ResearchResult


class SearchAgent:
    def __init__(self, search_tool: BaseTool):
        self.search_tool = search_tool

    def __call__(self, query: str):
        search_results = self.search_tool.invoke(input=query)
        if isinstance(search_results, dict):
            search_results = search_results["results"]

        results = []
        for search_result in search_results:
            results.append(
                ResearchResult(
                    url=search_result.get("url", None)
                    or search_result.get("link", None),
                    query=query,
                    title=search_result.get("title", "untitled"),
                    content=search_result.get("raw_content", None)
                    or search_result.get("snippet", None),
                )
            )
        return {"query_results": results}
