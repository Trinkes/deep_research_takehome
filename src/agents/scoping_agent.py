from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.deep_research_state import DeepResearchState

RESEARCH_DOCUMENT_DESCRIPTION = """
# Deep Research Document

## Metadata
- **Topic:**  
- **Objective:**  
- **Research Agent ID:**  
- **Date:**  
- **Version:**  

---

## 1. Research Definition
**Intent:** Establish the foundation of the research.  
**Agent Output:**  
- Problem definition  
- Research goals  
- Key questions or hypotheses  
- Assumptions and constraints  
- Expected deliverables  

---

## 2. Context and Background
**Intent:** Collect and summarize existing information relevant to the topic.  
**Agent Output:**  
- Overview of domain and terminology  
- Historical or conceptual background  
- Key frameworks, theories, or technologies  
- Known challenges or open problems  
- Summary of related work or prior studies  

---

## 3. Data & Sources
**Intent:** Identify and list the sources used for analysis.  
**Agent Output:**  
- Primary sources (academic papers, datasets, APIs, etc.)  
- Secondary sources (articles, web data, forums, etc.)  
- For each source:
  - Title or identifier  
  - URL (if applicable)  
  - Type (text, dataset, code, etc.)  
  - Relevance rating (1–5)  

---

## 4. Analysis
**Intent:** Process, extract, and reason about the gathered data.  
**Agent Output:**  
- Extracted facts, entities, or relationships  
- Statistical or semantic patterns  
- Contradictions or data gaps  
- Derived insights or hypotheses  
- Intermediate reasoning traces (optional)  

---

## 5. Synthesis
**Intent:** Combine findings into cohesive insights.  
**Agent Output:**  
- Core insights (with brief justification)  
- Conceptual model or explanation  
- Comparative analysis (if multiple viewpoints exist)  
- Visual or structural summary (if applicable)  

---

## 6. Implications
**Intent:** Interpret how findings impact the domain or objectives.  
**Agent Output:**  
- Practical or theoretical implications  
- Potential applications or use cases  
- Identified risks or limitations  
- Ethical, social, or legal considerations  

---

## 7. Conclusion
**Intent:** Summarize and finalize the research.  
**Agent Output:**  
- Final answer or main takeaway  
- Summary of supporting evidence  
- Confidence score (0–1)  
- Recommendations or next steps  

---

## 8. References
**Intent:** Log all materials referenced during research.  
**Agent Output:**  
- List of citations in standardized format (APA, MLA, etc.)  
- Source metadata (title, author, date, link)  
- Confidence or relevance rating per source  

---

## 9. Appendix (Optional)
**Intent:** Store auxiliary materials or reasoning context.  
**Agent Output:**  
- Raw extraction logs  
- Extended tables or datasets  
- Alternative hypotheses  
- Notes or observations  
"""


class ScopingResponse(BaseModel):
    needs_clarification: bool | None = Field(
        default=None,
        description="rather or not it we need more clarification from the user on the topic.",
    )
    answer: str = Field(description="Answer to give to be given to the user")
    document: str | None = Field(
        default=None,
        description="the full research_document_description document that will be use further to guide the research. "
        "This only needs to be filled in if no further clarification is needed"
        f"""<research_document_description>
                {RESEARCH_DOCUMENT_DESCRIPTION}
                </research_document_description>
                """,
    )


class ScopingAgent:
    def __init__(self, llm: BaseChatModel | None = None):
        self.llm: BaseChatModel = llm

    def __call__(self, state: DeepResearchState) -> dict:
        system_message = SystemMessage(
            content="""Your job is to prepare a **context document** that will guide a deep research process around a given topic.

**Output format:**
- `answer`: A brief message to the user. If clarification is needed, ask clear, concise, numbered questions. If ready to proceed, briefly confirm the research scope.
- `document`: The full research document structure (only if no clarification needed). Fill in each section with what should be researched based on the topic. If no further clarification is needed, a document must be provided
- `needs_clarification`: Set to true if you need any input from the user.

**Important:** Do NOT include the full research document structure in the `answer` field. Keep the answer concise and user-friendly."""
        )

        messages = [system_message] + list(state.messages)

        response: ScopingResponse = self.llm.with_structured_output(
            ScopingResponse
        ).invoke(messages)

        if response.needs_clarification:
            user_response = interrupt(response.answer)
            return {
                "messages": [
                    AIMessage(content=response.answer),
                    HumanMessage(content=user_response),
                ],
                "needs_research_clarification": True,
            }

        return {
            "messages": [AIMessage(content=response.answer)],
            "document": response.document,
            "needs_research_clarification": False,
        }
