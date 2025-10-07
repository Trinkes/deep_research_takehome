from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from typing import Annotated, Sequence

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.graph import START, END

load_dotenv()


class State(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list, description="Chat history")


graph_builder = StateGraph(State)


llm = init_chat_model("google_genai:gemini-2.0-flash")


def chatbot(state: State):
    return {"messages": [llm.invoke(state.messages)]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


deep_research_agent = graph_builder.compile()
