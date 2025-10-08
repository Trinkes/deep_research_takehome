# Deep Researcher Agent

A multi-agent system for conducting comprehensive internet research on any topic.

## Quick Start

### 1. Configure Environment

Copy the example environment file:

```shell
cp .env.example .env
```

Update `.env` with your API keys:

**LLM Provider (Required - choose one):**
- `GOOGLE_API_KEY` - Your Google API key (free tier available)
- `DEEPSEEK_API_KEY` - Your DeepSeek API key

**Model Configuration (Optional):**
- `MODEL_NAME` - Override the default model for your provider
  - Google default: `models/gemini-2.5-flash-lite`
  - DeepSeek default: `deepseek-chat`

**Search Configuration (Optional):**
- `TAVILY_API_KEY` - For enhanced search with full page content (falls back to DuckDuckGo if not provided)

**Observability (Optional):**
- `LANGSMITH_TRACING` - Set to `true` to enable LangSmith tracing
- `LANGSMITH_API_KEY` - Your LangSmith API key
- `LANGSMITH_PROJECT` - Your LangSmith project name

> **Note**: DuckDuckGo search uses snippets instead of full page content since it doesn't provide page content directly.

### 2. Run the Application

Choose one of the following options:

**Option A: Docker Compose** (Recommended)

```shell
docker compose up
```

**Option B: UV Package Manager**

```shell
uv sync
uv run langgraph dev
```

### 3. Access the Studio

Open the LangGraph Studio at:

```
https://smith.langchain.com/studio/?baseUrl=http://localhost:2024
```

> [!IMPORTANT]
> When using Docker, always use `baseUrl=http://localhost:2024` (not `http://0.0.0.0:2024`) to ensure proper
> connectivity.

## Agent Architecture

The system consists of three specialized agents working together:

### Research Agent (`research_agent`)

Performs focused internet searches on a single topic.

- **Purpose**: Deep dive into individual research topics
- **Control**: `max_queries_per_topic` - Limits the number of searches per topic

### Research Agent Orchestrator (`research_agent_orchestrator`)

Manages the overall research process and coordinates multiple Research Agents.

- **Purpose**: Breaks down research into topics and delegates to Research Agents
- **Control**: `max_generated_topics` - Limits the number of topics extracted from research context

### Deep Research Agent (`deep_research_agent`)

Top-level agent that gathers requirements and orchestrates comprehensive research.

- **Purpose**: Collects user context, generates research document, and delegates to orchestrator
- **Control**:
    - `max_generated_topics` - Maximum topics to research
    - `max_queries_per_topic` - Maximum searches per topic

## API Usage

### Basic Usage

```python
from langchain_core.messages import HumanMessage
from src.deep_research_agent import DeepResearchAgent
from src.deep_research_state import DeepResearchState
from main import deep_research_agent

# Initialize the agent
graph = deep_research_agent()
agent = DeepResearchAgent(graph)

# Create state with your research query
state = DeepResearchState(
    messages=[HumanMessage(content="Research the impact of AI on healthcare")]
)

# Perform research
result = agent.perform_research(state)
```

### Configuration

Control research depth and execution behavior:

```python
# Configure research parameters
state = DeepResearchState(
    messages=[HumanMessage(content="Your research topic")],
    max_generated_topics=4,  # Number of topics to extract (default: 4)
    max_queries_per_topic=2,  # Searches per topic (default: 2)
)

result = agent.perform_research(
    state,
    config={
        "recursion_limit": 50,  # Max node executions (default: 25)
    }
)
```

### Custom Output Format

To customize the research report structure, modify the output format in `main.py`:

```python
# In main.py, update the research_agent_orchestrator function:
def research_agent_orchestrator() -> CompiledStateGraph:
    llm = create_llm()
    research_agent_graph = research_agent()

    custom_format = """
    # Your Custom Report Structure
    ## Section 1
    ## Section 2
    """

    return (
        ResearchAgentOrchestratorGraphBuilder()
        .with_llm(llm)
        .with_research_graph(research_agent_graph)
        .with_output_structure(custom_format)  # Pass your custom format here
        .build_graph()
    )
```

## Notes

- All tests were done using `deepseek-chat` model
- Google Gemini option added to provide a free way of running the project

## Out of scope

- Provide execution insights (e.g., showing URLs searched in real-time) 
- better error handling ex: 
  - use a fallback model if the main one fails
  - show an error to the user
- Experiment with different models on each step and optimize for quality or pricing

