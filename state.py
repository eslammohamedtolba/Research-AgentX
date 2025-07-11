from typing import TypedDict, Literal, List
from pydantic import BaseModel, Field

class ResearchState(TypedDict):
    """The complete state of the research agent."""
    query: str
    web_results: List[str]
    arxiv_results: List[str]
    rag_results: List[str]
    final_answer: str

class Decision(BaseModel):
    """The decision model for the research strategist."""
    next: Literal[
        "web_search_node",
        "fetch_arxiv_node",
        "rag_search_node",
        "Synthesizer_node"
    ] = Field(description="The next tool to use based on the research plan.")
    
    reason: str = Field(description="A brief explanation for why this tool was chosen.")