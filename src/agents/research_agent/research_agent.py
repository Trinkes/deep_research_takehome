from langchain.prompts import Prompt
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from src.agents.research_agent.research_state import ResearchState, ResearchResult
from src.base_agent import BaseAgent


class ResearchAgentResponse(BaseModel):
    queries: list[str] = Field(
        default_factory=list, description="list of search queries"
    )


class ResearchAgent(BaseAgent):
    name = "research_agent"

    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm: BaseLanguageModel = llm

    def __call__(self, state: ResearchState) -> dict:
        prompt = Prompt.from_template("""
        you are a research agent and your job is to take a topic and create a list of internet search queries
        
        <queries_already_made>
        {queries}
        </queries_already_made>
        
        <research_topic>
        {research_topic}
        </research_topic>
        """)
        result: ResearchAgentResponse = self.llm.with_structured_output(
            ResearchAgentResponse
        ).invoke(
            prompt.format(
                queries=state.performed_queries, research_topic=state.research_topic
            )
        )
        results = []
        for query in result.queries:
            # search = TavilySearch(max_results=1, include_raw_content=True, )
            search = DuckDuckGoSearchResults(output_format="list")

            search_results = search.invoke(input=query)["results"]
            for search_result in search_results:
                results.append(
                    ResearchResult(
                        url=search_result["url"],
                        query=query,
                        title=search_result["title"],
                        content=search_result.get("raw_content", None)
                        or search_result.get("snippet", None),
                    )
                )

        return {"performed_queries": result.queries, "query_results": results}
