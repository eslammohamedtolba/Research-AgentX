from typing import TypedDict, List, Annotated, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class ResearchState(TypedDict):
    """The complete state of the research agent."""
    
    # The full conversation history
    messages: Annotated[Sequence, add_messages]
    
    # The query used by tools, which can be refined
    refined_query: str
    
    # Consolidated list of all documents that have passed the grader
    related_documents: List[str]
    
    # Raw documents from a search node, awaiting grading
    sources: List[str]
    
    # Counters for the strategist's status report
    refinements_web_used: int
    refinements_arxiv_used: int
    refinements_rag_used: int

    newly_added_count: int
    
    # The strategist's decision on the next tool to run
    active_tool: str


class SourceGrade(BaseModel):
    """Boolean value to check if the document is related to the question or not"""
    related: bool = Field(description="Document is related to the question? True/False")
