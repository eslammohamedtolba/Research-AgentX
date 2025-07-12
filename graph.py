from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# Import from our local project files
from state import ResearchState
from nodes import (
    research_strategist,
    web_search_node,
    arxiv_search_node,
    rag_search_node,
    grade_and_filter_node,
    refine_query_node,
    synthesizer_node,
    route_after_strategist,
    route_to_next_tool
)

def create_graph():
    """Builds and compiles the LangGraph agent."""
    graph = StateGraph(ResearchState)
    
    # Add all the nodes
    graph.add_node("Research_Strategist", research_strategist)
    graph.add_node("refine_query", refine_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("arxiv_search", arxiv_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("grade_and_filter", grade_and_filter_node)
    graph.add_node("synthesize", synthesizer_node)
    
    # Set the entry point
    graph.set_entry_point("Research_Strategist")
    
    # Add the first conditional edge from the strategist
    graph.add_conditional_edges(
        "Research_Strategist",
        route_after_strategist,
        {
            "refine_query": "refine_query",
            "web_search": "web_search",
            "arxiv_search": "arxiv_search",
            "rag_search": "rag_search",
            "synthesize": "synthesize"
        }
    )
    
    # Add the second conditional edge from the refinement node
    graph.add_conditional_edges(
        "refine_query", 
        route_to_next_tool,
        {
            "web_search": "web_search",
            "arxiv_search": "arxiv_search",
            "rag_search": "rag_search"
        }
    )
    
    # Define edges
    graph.add_edge("web_search", "grade_and_filter")
    graph.add_edge("arxiv_search", "grade_and_filter")
    graph.add_edge("rag_search", "grade_and_filter")
    graph.add_edge("grade_and_filter", "Research_Strategist")
    
    graph.add_edge("synthesize", END)
    
    # Create memory and compile the graph
    memory = MemorySaver()
    
    return graph.compile(checkpointer=memory)