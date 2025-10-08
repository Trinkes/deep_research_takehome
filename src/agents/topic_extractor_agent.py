from langchain.prompts import Prompt
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)


class TopicExtractorResponse(BaseModel):
    topics: list[str] = Field(
        default_factory=list,
        description="List of specific research topics to investigate",
    )


class TopicExtractorAgent:
    name = "topic_extractor_agent"

    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm: BaseLanguageModel = llm

    def __call__(self, state: OrchestratorResearchState):
        prompt = Prompt.from_template("""You are a research topic extraction and planning agent. Your goal is to analyze a research description and identify specific, focused topics that need to be researched.

# Research Description
<research_description>
{research_description}
</research_description>

# Already Researched Topics
<researched_topics>
{researched_topics}
</researched_topics>

# Current Research Results
<results>
{results}
</results>

# Your Task
1. Analyze the research description to understand what needs to be researched
2. Review the already researched topics to avoid duplicates
3. Examine the current research results to identify:
   - What questions have been answered
   - What information is still missing
   - What gaps exist in the current research
4. Extract or generate 1-3 new, specific topics that should be researched next

# Guidelines
- Each topic should be focused and specific (not too broad)
- Topics should be directly relevant to answering the research description
- Avoid topics that have already been researched
- If current results are incomplete or raise new questions, suggest topics to fill those gaps
- If the research appears complete based on the results, you may return an empty list

Examples of good topics:
- "Python async/await performance benchmarks"
- "FastAPI middleware implementation patterns"
- "React Server Components rendering lifecycle"

Examples of bad topics (too broad):
- "Python"
- "Web development"
- "Machine learning"
""")
        if len(state.searched_topics) == 0:
            researched_topics = ["No results yet"]
        else:
            researched_topics = state.searched_topics

        response: TopicExtractorResponse = self.llm.with_structured_output(
            TopicExtractorResponse
        ).invoke(
            prompt.format(
                research_description=state.research_description,
                researched_topics=", ".join(researched_topics),
                results="\n".join(f"- {result}" for result in state.results),
            )
        )
        topics = response.topics
        exceeding_topics = (
            len(topics) + len(state.searched_topics) - state.max_generated_topics
        )
        if exceeding_topics > 0:
            topics = topics[:-exceeding_topics]
        return {"searched_topics": topics}
