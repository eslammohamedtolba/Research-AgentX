from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 

# Import from our local project files
from state import ResearchState
from nodes import (
    refine_query_node,
    web_search_node,
    arxiv_search_node,
    rag_search_node,
    grade_and_filter_node,
    route_after_grading,
    route_after_refine,
    synthesizer_node,
)

def create_graph():
    """Builds and compiles the LangGraph agent."""
    
    graph = StateGraph(ResearchState)
    
    # Add all the nodes
    graph.add_node("refine_query", refine_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("arxiv_search", arxiv_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("grade_and_filter", grade_and_filter_node)
    graph.add_node("synthesize", synthesizer_node)
    
    # Set the entry point
    graph.set_entry_point("refine_query") # Start with the first search tool
    
        # Conditional edge after refining the query
    graph.add_conditional_edges(
        "refine_query",
        route_after_refine,
        {
            "web": "web_search",
            "arxiv": "arxiv_search",
            "rag": "rag_search"
        }
    )
    
    # Define edges
    graph.add_edge("web_search", "grade_and_filter")
    graph.add_edge("arxiv_search", "grade_and_filter")
    graph.add_edge("rag_search", "grade_and_filter")
    
    # Central conditional edge after grading documents
    graph.add_conditional_edges(
        "grade_and_filter",
        route_after_grading,
        {
            "need_refine": "refine_query",
            "arxiv": "arxiv_search",
            "rag": "rag_search",
            "synthesize": "synthesize"
        }  
    )
    
    graph.add_edge("synthesize", END)
    
    # Create connection and memory
    sqlite_conn = sqlite3.connect("db.sqlite", check_same_thread = False)
    memory = SqliteSaver(conn = sqlite_conn)
    
    # compile the graph
    return graph.compile(checkpointer=memory)
