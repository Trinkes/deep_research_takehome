import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph.state import CompiledStateGraph

from src.agents.orchestrator.orchestrator_research_agent_graph_builder import (
    ResearchAgentOrchestratorGraphBuilder,
)
from src.agents.research_agent.research_agent_builder import ResearchAgentBuilder
from src.deep_research_agent import DeepResearchAgent
from src.deep_research_graph_builder import DeepResearchGraphBuilder
from src.deep_research_state import DeepResearchState


def deep_research_agent() -> CompiledStateGraph:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    orchestrator = research_agent_orchestrator()
    return (
        DeepResearchGraphBuilder()
        .with_llm(llm)
        .with_orchestrator(orchestrator)
        .build_graph()
    )


def research_agent_orchestrator() -> CompiledStateGraph:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    research_agent_graph = research_agent()
    return (
        ResearchAgentOrchestratorGraphBuilder()
        .with_llm(llm)
        .with_research_graph(research_agent_graph)
        .build_graph()
    )


def research_agent() -> CompiledStateGraph:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    if os.getenv("TAVILY_API_KEY"):
        search = TavilySearch(
            max_results=1,
            include_raw_content=True,
        )
    else:
        search = DuckDuckGoSearchResults(output_format="list")

    return (
        ResearchAgentBuilder().with_llm(llm=llm).with_search_tool(search).build_graph()
    )


if __name__ == "__main__":
    load_dotenv()
    graph = deep_research_agent()
    agent = DeepResearchAgent(graph)
    state = DeepResearchState(
        messages=[
            HumanMessage(
                content="I want to do some research about the weather forecast and its impact in the portuguese economy"
            )
        ]
    )
    print(agent.perform_research(state, config={"configurable": {"thread_id": "1"}}))
