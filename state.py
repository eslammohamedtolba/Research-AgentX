from typing import TypedDict, Literal, List
from pydantic import BaseModel, Field

class ResearchState(TypedDict):
    """The complete state of the research agent."""
    
    # The original user query
    query: str
    
    # The query used by tools, which can be refined
    refined_query: str
    
    # Consolidated list of all documents that have passed the grader
    related_documents: List[str]
    
    # Raw documents from a search node, awaiting grading
    sources: List[str]
    
    # Counters for the strategist's status report
    refinements_used: int
    web_count: int
    arxiv_count: int
    rag_count: int
    
    # The strategist's decision on whether to refine the query before the next node
    should_refine:bool
    # The strategist's decision on the next tool to run
    active_tool: str
    
    final_answer: str

class StrategistDecision(BaseModel):
    """The decision model for the research strategist."""
    
    next: Literal[
        "web_search",
        "arxiv_search",
        "rag_search",
        "synthesize"
    ] = Field(description="The next tool to use in the research plan.")
    
    refine: bool = Field(
        description="Set to True if the current query is not yielding good results and needs to be rephrased before the next action."
    )
    
    reason: str = Field(description="A brief explanation for why this tool was chosen.")
    
class SourceGrade(BaseModel):
    """Boolean value to check if the document is related to the question or not"""
    related: bool = Field(description="Document is related to the question? True/False")
