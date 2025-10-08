# Deep Researcher Agent

A multi-agent system for conducting comprehensive internet research on any topic.

## Quick Start

### 1. Configure Environment

Copy the example environment file:

```shell
cp .env.example .env
```

Update `.env` with your API keys:
- **Required**: `GOOGLE_API_KEY` - Your Google API key
- **Optional**: `TAVILY_API_KEY` - For enhanced search (falls back to DuckDuckGo if not provided)
- **Optional**: `LANGSMITH_*` - For LLM execution tracing in LangSmith

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
> When using Docker, always use `baseUrl=http://localhost:2024` (not `http://0.0.0.0:2024`) to ensure proper connectivity.

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

