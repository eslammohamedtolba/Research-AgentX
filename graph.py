from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import from our local project files
from state import ResearchState
from nodes import (
    research_strategist,
    web_search_node,
    fetch_arxiv_node,
    rag_search_node,
    synthesizer_node
)

def create_graph():
    """Builds and compiles the LangGraph agent."""
    graph = StateGraph(ResearchState)
    
    # Add all the nodes
    graph.add_node("Research_Strategist_node", research_strategist)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("fetch_arxiv_node", fetch_arxiv_node)
    graph.add_node("rag_search_node", rag_search_node)
    graph.add_node("Synthesizer_node", synthesizer_node)
    
    # Set the entry point and define edges
    graph.set_entry_point("Research_Strategist_node")
    graph.add_edge("web_search_node", "Research_Strategist_node")
    graph.add_edge("fetch_arxiv_node", "Research_Strategist_node")
    graph.add_edge("rag_search_node", "Research_Strategist_node")
    graph.add_edge("Synthesizer_node", END)
    
    # Create memory and compile the graph
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)