from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

# from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from src.agents.orchestrator.orchestrator_research_agent_graph_builder import (
    ResearchAgentOrchestratorGraphBuilder,
)
from src.agents.research_agent.research_agent_builder import ResearchAgentBuilder
from src.deep_research_agent import DeepResearchAgent
from src.deep_research_graph_builder import DeepResearchGraphBuilder
from src.deep_research_state import DeepResearchState


def deep_research_agent() -> CompiledStateGraph:
    # llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    llm = ChatDeepSeek(model="deepseek-chat", max_tokens=4000, temperature=0)
    return DeepResearchGraphBuilder(llm).build_graph()


def research_agent_orchestrator() -> CompiledStateGraph:
    #     llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    llm = ChatDeepSeek(model="deepseek-chat", max_tokens=4000, temperature=0)
    research_agent_graph = research_agent()
    return (
        ResearchAgentOrchestratorGraphBuilder()
        .with_llm(llm)
        .with_research_graph(research_agent_graph)
        .build_graph()
    )


def research_agent() -> CompiledStateGraph:
    #     llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    llm = ChatDeepSeek(model="deepseek-chat", max_tokens=4000, temperature=0)
    return ResearchAgentBuilder().with_llm(llm=llm).build_graph()


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
