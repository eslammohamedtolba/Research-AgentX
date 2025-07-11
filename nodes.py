import arxiv
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage

# Import from our local project files
from state import ResearchState, Decision
from config import llm, search_tool, embedding_function
from rag_setup import get_retriever

# Initialize the retriever when this module is loaded
retriever = get_retriever(embedding_function)

def research_strategist(state: ResearchState) -> str:
    """The central 'brain' of the agent."""
    
    structured_llm = llm.with_structured_output(Decision)
    
    system_prompt = (
        "You are an expert research strategist. Your job is to create a step-by-step plan to answer a user's query."
        "You have the following tools available:\n"
        "1. search_web: Use for current events, general knowledge, or broad topics.\n"
        "2. fetch_arxiv: Use for accessing scientific papers and technical research on ArXiv.\n"
        "3. rag_search: Use to search your internal database of specialized academic papers.\n"
        "4. Synthesize: Use this ONLY when you have gathered enough information from the other tools to provide a complete answer.\n\n"
        "Review the user's query and the current results. Decide on the single best next step. "
        "Do not use a tool if it has already been used and provided results. "
        "Your goal is to gather comprehensive information before calling the Synthesizer."
    )
    
    human_prompt = (
        f"Query: {state['query']}\n\n"
        f"Current Web Results: {bool(state.get('web_results'))}\n"
        f"Current ArXiv Results: {bool(state.get('arxiv_results'))}\n"
        f"Current RAG Results: {bool(state.get('rag_results'))}\n\n"
        "Based on this, what is the next best step?"
    )
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=human_prompt)
    ]
    
    response = structured_llm.invoke(messages)
    
    return Command(
        goto=response.next,
        update=state
    )

def web_search_node(state: ResearchState) -> dict:

    web_results = search_tool.invoke([{"query": state["query"]}])
    
    return {
        "web_results": [res['content'] for res in web_results]
    }

def fetch_arxiv_node(state: ResearchState) -> dict:
    
    search = arxiv.Search(
        query=state['query'], 
        max_results=3, 
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    summaries = [result.summary.replace("\n", " ") for result in search.results()]
    
    return {
        "arxiv_results": summaries
    }

def rag_search_node(state: ResearchState) -> dict:

    retrieved_docs = retriever.invoke(state["query"]) or []
    
    return {
        "rag_results": [doc.page_content for doc in retrieved_docs]
    }

def synthesizer_node(state: ResearchState) -> dict:

    context = ""
    if state.get("web_results"): context += "Web Search Results:\n" + "\n".join(state["web_results"]) + "\n\n"
    if state.get("arxiv_results"): context += "ArXiv Paper Summaries:\n" + "\n".join(state["arxiv_results"]) + "\n\n"
    if state.get("rag_results"): context += "Relevant Academic Paper Chunks:\n" + "\n".join(state["rag_results"]) + "\n\n"
    
    system_prompt = (
        "You are an expert research assistant. Your goal is to provide a comprehensive and "
        "well-structured answer to the user's query based *only* on the provided context from web, ArXiv, and academic paper searches. "
        "Synthesize the information from all sources into a single, coherent response. "
        "If the context is empty or insufficient, state that you could not find enough information."
    )
    
    user_prompt = f"Query: {state['query']}\n\nContext:\n{context}"
    
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_answer": response.content
    }
    

