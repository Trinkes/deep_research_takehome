# Deep Researcher Agent

## How to run

1. copy `.env.example` file to `.env`

```shell
cd path_to_project_root_folder
cp .env.example .env
```

2. Update `.env` file
    1. Fill in `GOOGLE_API_KEY` variable
    2. (Optional) Fill in `TAVILY_API_KEY`. Duck Duck Go search will be used if `TAVILY_API_KEY` is not filled in. since
       DuckDuckGo doesn't provide the page's content, the snippet field will be used
    3. (Optional) Fill in `LANGSMITH*` keys to be able to trace the agent llm execution
3. then we have 2 options:
    1. using docker compose
    ```shell
    docker compose up
    ```
    2. using uv
    ```shell
    uv sync
    uv run langgraph dev
    ```
4. open https://smith.langchain.com/studio/?baseUrl=http://localhost:2024

> [!IMPORTANT]
> When running through docker, make sure you use the URL provided above (step 4). If you use the url provided by the
> langgraph dev command, it won't work since it will use `?baseUrl=http://0.0.0.0:2024` instead of
`?baseUrl=http://localhost:2024`

## Available agents

There are three different agents available each one of them has its own scope:

### Research Agent (`research_agent`)

This agent's job is to get in a topic and make as many internet searches as needed about the provided topic.
This agent has a state variable to control the max number of searches per topic.

### Research Agent Orchestrator (`research_agent_orchestrator`)

Agent that manages the overall research.
It extracts topics from the provided research description (context) and uses `Research Agent` to search for a specific
topic.
It will extract as many topics from the context as it needs to create the research.
This agent has a state variable to control the max number of topics extracted from the research document.

### Deep Research Agent (`deep_research_agent`)

This agent will request more context from the user if needed and produce a detailed document about the research to be
performed. Then passes this document to `Research Agent Orchestrator` for him to proceed with the research.
State variables `max_generated_topics` and `max_queries_per_topic` to control the research depth

