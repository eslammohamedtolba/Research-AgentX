import arxiv
import random
import time
from langchain_core.messages import SystemMessage, HumanMessage

# Import from our local project files
from state import ResearchState, StrategistDecision, SourceGrade
from config import llm, search_tool, embedding_function
from rag_setup import get_retriever

# Initialize the retriever when this module is loaded
retriever = get_retriever(embedding_function)

# Initialize the max number of refinements the strategist can use
MAX_REFINEMENTS = 3

# Helper Function for the Strategist's Prompt
def create_status_report(state: ResearchState) -> str:
    
    """Creates a formatted status report for the strategist's prompt."""
    total_docs = len(state.get('related_documents', []))
    refinements = state.get('refinements_used', 0)

    def get_source_status(count_key: str):
        count = state.get(count_key, 0)
        # Check if the key exists (meaning the tool has run and been graded)
        if count is None:
            return "Not chosen yet"
        if count == 0:
            return "Chosen, but didn't find any information"
        return str(count)

    return (
        f"Refinements Used: {refinements}/{MAX_REFINEMENTS}\n"
        f"Total Relevant Documents Found: {total_docs}\n\n"
        "Source Breakdown (Relevant Documents):\n"
        f"- Web Search: {get_source_status('web_count')}\n"
        f"- ArXiv Search: {get_source_status('arxiv_count')}\n"
        f"- RAG Search: {get_source_status('rag_count')}"
    )

def research_strategist(state: ResearchState) -> ResearchState:
    """The central 'brain' of the agent."""
    
    # Initialize refined_query if it doesn't exist
    if not state.get('refined_query'):
        state['refined_query'] = state['query']
    
    time.sleep(10)
    
    structured_llm = llm.with_structured_output(StrategistDecision)
    
    system_prompt = (
        "You are an expert research strategist. Your job is to create a plan to answer a user's query by making a two-part decision:\n"
        "1.  **`refine` (True/False)**: First, decide if the current query needs to be rephrased to get better results. Set this to `True` only if a search has failed.\n"
        "2.  **`next` (Tool Name)**: Second, choose the next tool to run (`web_search`, `arxiv_search`, `rag_search`, or `synthesize`).\n\n"
        
        "### Decision-Making Rules:\n"
        "1.  **Search First**: If any tool reports 'Not chosen yet', your `next` action MUST be to choose one of those tools. You MUST set `refine` to `False`.\n"
        "2.  **Handle Failure**: If all tools have been tried but one or more reported 'Chosen, but didn't find any information' AND you have refinements left, you MUST set `refine` to `True` and set `next` to the tool that failed (e.g., `web_search`).\n"
        "3.  **Synthesize**: Once all tools have been tried successfully, or if you have run out of refinements, you MUST choose `synthesize` as your `next` action and set `refine` to `False`."
    )
    
    status_report = create_status_report(state)
    print("\n\n\n", "report: ", status_report, "\n\n\n")
    
    human_prompt = f"Original Query: {state['query']}\n\nReport: {status_report}\n\nBased on the rules, create a plan. What is your two-part decision for `refine` and `next`?"
    
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=human_prompt)
    ]
    
    time.sleep(10)
    
    response = structured_llm.invoke(messages)
    
    
    return {
        "should_refine": response.refine,
        "active_tool": response.next
    }

def route_after_strategist(state: ResearchState):
    """Decides whether to refine the query or proceed directly to the next tool."""
    
    time.sleep(10)
    
    if state.get("should_refine"):
        return "refine_query"
    else:
        return state.get('active_tool')

def route_to_next_tool(state: ResearchState):
    """Routes to the next tool planned by the strategist."""
    
    time.sleep(10)
    
    return state.get('active_tool')

# Helper function to create query variations
def refine_query_node(state: ResearchState) -> ResearchState:
    """Uses an LLM to create a variation of the original query."""
    
    query = state['refined_query']
    
    time.sleep(10)
    
    refine_prompt = f"Rephrase the following research query to find different but related results. Return only the new query. Original query: {query}"
    new_query = llm.invoke(refine_prompt).content
    
    time.sleep(10)
    
    return {
        'refined_query':new_query,
        'refinements_used': state.get('refinements_used', 0) + 1,
        'should_refine': False
    }

def web_search_node(state: ResearchState) -> ResearchState:

    results = search_tool.invoke({"query": state['refined_query']})
    
    docs = [res['content'] for res in results]
    
    time.sleep(10)
    
    return {
        "sources": docs, 
        "active_tool": "web"
        }

def arxiv_search_node(state: ResearchState) -> ResearchState:
    
    # Randomly pick the sort criterion to provide diversity
    sort_criterion = random.choice([
        arxiv.SortCriterion.Relevance, 
        arxiv.SortCriterion.SubmittedDate
    ])
    
    search = arxiv.Search(
        query=state['refined_query'], 
        max_results=3,
        sort_by=sort_criterion
    )
    summaries = [result.summary.replace("\n", " ") for result in search.results()]
    
    time.sleep(10)
    
    return {
        "sources": summaries, 
        "active_tool": "arxiv"
    }

def rag_search_node(state: ResearchState) -> ResearchState:

    retrieved_docs = retriever.invoke(state['refined_query']) or []
    
    time.sleep(10)
    
    return {
        "sources": [doc.page_content for doc in retrieved_docs],
        "active_tool": "rag"
    }

def grade_and_filter_node(state: ResearchState) -> ResearchState:
    """Grades a list of documents and returns only the relevant ones."""
    
    query = state['query'] # Always grade against the original query
    documents = state['sources']
    active_tool = state['active_tool']
    
    grader = llm.with_structured_output(SourceGrade)
    relevant_docs = []
    for doc in documents:
        prompt = f"Is the following document relevant to the user's question? Document:\n\n{doc}\n\nQuestion: {query}"
        response = grader.invoke(prompt)
        if response.related:
            relevant_docs.append(doc)
        time.sleep(4)
        
    # Append relevant docs to the main list
    current_related = state.get('related_documents', [])
    updated_related_docs = current_related + relevant_docs
            
    # Update the correct counter based on which tool ran
    count_key = f"{active_tool}_count"
    current_count = state.get(count_key) or 0
    updated_count = current_count + len(relevant_docs)
    
    return {
        "related_documents": updated_related_docs,
        count_key: updated_count,
        "sources": [] # Clear the temporary sources list
    }
    
def synthesizer_node(state: ResearchState) -> ResearchState:

    context = "\n\n---\n\n".join(state.get('related_documents', []))
    
    if not context:
        return {"final_answer": "After running the research process, no relevant documents were found to answer the query."}
    
    time.sleep(10)
    
    prompt = f"Based ONLY on the following documents, provide a comprehensive and well-structured answer to the user's original query.\n\nOriginal Query: {state['query']}\n\nDocuments:\n{context}"
    response = llm.invoke(prompt)
    
    time.sleep(10)
    
    return {"final_answer": response.content}
    


