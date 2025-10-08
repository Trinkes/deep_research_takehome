from langchain.prompts import Prompt
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from src.agents.research_agent.research_state import ResearchState


class ResearchAgentResponse(BaseModel):
    queries: list[str] = Field(
        default_factory=list, description="list of search queries"
    )


class QueryExtractor:
    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm: BaseLanguageModel = llm

    def __call__(self, state: ResearchState) -> dict:
        prompt = Prompt.from_template("""
You are a research agent specializing in generating effective internet search queries.

Your task is to analyze the research topic and create a list of targeted search queries that will gather comprehensive information about the topic.

Guidelines:
1. Generate 3-5 diverse search queries that cover different aspects of the topic
2. Make queries specific and focused rather than broad and generic
3. Avoid duplicating queries that have already been made
4. Use natural language that search engines can effectively process
5. IMPORTANT: Only return an empty list if you have already made 5+ queries about this topic and are confident all aspects have been thoroughly covered

<queries_already_made>
{queries}
</queries_already_made>

<research_topic>
{research_topic}
</research_topic>

If the queries_already_made list is empty, you MUST generate initial search queries to start the research. Generate search queries now.
        """)
        result: ResearchAgentResponse = self.llm.with_structured_output(
            ResearchAgentResponse
        ).invoke(
            prompt.format(queries=state.queries, research_topic=state.research_topic)
        )
        queries = result.queries
        number_exceeding_queries = len(state.queries) + len(queries) - state.max_queries
        if number_exceeding_queries > 0:
            queries = queries[:-number_exceeding_queries]

        return {"queries": queries}
