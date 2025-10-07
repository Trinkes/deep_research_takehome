import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.deep_research_agent import DeepResearchAgent
from src.deep_research_graph_builder import DeepResearchGraphBuilder
from src.deep_research_state import DeepResearchState


def deep_research_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    return DeepResearchGraphBuilder(llm).build_graph()


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
    print(asyncio.run(agent.perform_research(state)))
