from langchain_core.tools import BaseTool

from src.agents.research_agent.research_state import ResearchResult


class SearchAgent:
    def __init__(self, search_tool: BaseTool):
        self.search_tool = search_tool

    def __call__(self, query: str):
        search_results = self.search_tool.invoke(input=query)

        if isinstance(search_results, dict):
            search_results = search_results.get("results", [])

        if not search_results:
            return {"query_results": []}

        results = []
        for search_result in search_results:
            if not isinstance(search_result, dict):
                continue

            url = search_result.get("url") or search_result.get("link") or ""
            title = search_result.get("title") or "Untitled"
            content = (
                search_result.get("raw_content")
                or search_result.get("content")
                or search_result.get("snippet")
            )

            if not content:
                continue

            results.append(
                ResearchResult(url=url, query=query, title=title, content=content)
            )

        return {"query_results": results}
