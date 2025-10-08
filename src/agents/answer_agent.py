from langchain.prompts import Prompt
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from src.agents import DEFAULT_RESEARCH_OUTPUT_FORMAT
from src.agents.orchestrator.orchestrator_research_state import (
    OrchestratorResearchState,
)


class AnswerResponse(BaseModel):
    report: str = Field(
        description="The comprehensive research report synthesizing all research items and context into the specified output format"
    )


class AnswerAgent:
    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
        output_format: str = DEFAULT_RESEARCH_OUTPUT_FORMAT,
    ):
        self.output_format = output_format
        self.llm: BaseLanguageModel = llm

    def __call__(self, state: OrchestratorResearchState):
        prompt = Prompt.from_template("""
You are an expert research writer tasked with creating a comprehensive research report.

Your goal is to synthesize all the research items below into a cohesive, well-structured report that addresses the research context. You must:

1. Analyze and integrate information from all research items
2. Organize the findings logically and coherently
3. Cite sources using the titles provided in each research item
4. Identify patterns, connections, and key insights across the research
5. Present a balanced view when multiple perspectives exist
6. Follow the specified output format precisely

<context>
{context}
</context>

<research_items>
{research}
</research_items>

<output_format>
{output_format}
</output_format>

Generate a comprehensive research report based on the above information.
        """)

        research_results = []
        for result in state.results:
            for query_result in result.query_results:
                content = f"""
<research_item>
<title>{query_result.title}</title>
<content>
{query_result.content}
</content>
</research_item>
"""
                research_results.append(content)
        if len(research_results) == 0:
            research_results.append("No research results available to generate a report.")

        response: AnswerResponse = self.llm.with_structured_output(
            AnswerResponse
        ).invoke(
            prompt.format(
                context=state.research_description,
                research="\n".join(research_results),
                output_format=self.output_format,
            )
        )

        return {"research_report": response.report}
