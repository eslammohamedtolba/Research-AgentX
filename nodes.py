import arxiv
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage

# Import from our local project files
from state import ResearchState, SourceGrade
from config import llm, search_tool, embedding_function
from rag_setup import get_retriever

# Initialize the retriever when this module is loaded
retriever = get_retriever(embedding_function)

# Initialize the max number of refinements the strategist can use
MAX_REFINEMENTS = 2

# Helper function to create query variations
def refine_query_node(state: ResearchState) -> ResearchState:
    """
    Uses an LLM to refine the query based on the conversation history and the failing tool.
    """

    active_tool = state.get('active_tool', 'web')
    
    # This new prompt instructs the LLM on how to handle different query types.
    prompt = (
        f"You are a hyper-competent, expert-level Query Strategist. Your sole mission is to dissect a conversation and formulate the most effective search query possible for a specific tool: '{active_tool}'.\n"
        "Your output MUST be ONLY the raw query string, with no extra text, explanations, or quotes.\n\n"
        "--- ANALYSIS AND REFINEMENT PROCESS ---\n"
        "1.  **Identify the Core User Goal:** Look at the most recent user message. What is the fundamental thing they want to *find* or *know*? Are they asking for paper titles, explanations, comparisons, code, etc.? The nature of their goal is critical.\n\n"
        "2.  **Resolve Context from Full History:** Read the ENTIRE conversation history, including the AI's previous responses. Users often use vague terms like 'that', 'those topics', or 'what you said'. Your most important task is to replace these vague terms with the specific, concrete concepts from the preceding conversation. \n"
        "    * **Example:** If the AI previously mentioned 'Multi-Agent Reinforcement Learning (MARL)' and the user then asks 'give me paper names for those topics', your query must explicitly include 'Multi-Agent Reinforcement Learning' or 'MARL'.\n\n"
        "3.  **Handle Failure and Re-attempt:** If the query is a refinement because a previous search failed, you MUST NOT just make a minor tweak. Radically rephrase the query. Broaden it, narrow it, or use completely different synonyms and keywords to overcome the previous failure.\n\n"
        "4.  **Synthesize the Final, Standalone Query (Calibrating for Intent):** Combine the core goal and resolved context. Critically, you must calibrate the query's depth and detail based on the inferred user intention, even if they don't use words like 'detailed'.\n"
        "    * **Assess the Implicit Need for Detail:** Analyze the nature of the user's query. Is it a simple factual question (e.g., 'who created X?', 'when was Y released?')? Or is it a complex, open-ended question that requires synthesizing information (e.g., 'how does X work?', 'what is the impact of Y?', 'compare X and Z')?\n"
        "    * **For Complex/Open-Ended Intent:** If the user's intention is clearly to understand a process, a relationship between concepts, or the implications of a topic, your query MUST be more comprehensive. It should include multiple facets of the topic to find rich documents capable of supporting a nuanced answer.\n"
        "    * **Detail Example:** If the conversation is about LLMs and the user asks, 'How are they being used in scientific research', this implies a need for a detailed overview. A good query would be: `applications of large language models in scientific discovery biology drug design materials science`. This is far more effective than just `LLMs in scientific research`.\n"
        "    * **For Simple/Factual Intent:** If the intention is to find a specific fact, the query should be concise and targeted. For 'Who created the Transformer architecture?', the query `who created the transformer neural network architecture` is sufficient.\n\n"
        "5.  **Tool-Specific Optimization:** Fine-tune the query for the '{active_tool}' tool:\n"
        "    * `web_search`: Can be a natural language question or keywords.\n"
        "    * `arxiv_search`: Should be formal and academic. Use precise technical terms.\n"
        "    * `rag_search`: Should use terminology very specific to the likely content of the local database (e.g., machine learning paper abstracts).\n\n"
        "--- CONVERSATION HISTORY ---\n"
        f"{state['messages']}\n\n"
        "Final, optimized query string:"
    )
    
    new_query = llm.invoke(prompt)
    
    refinement_key = f"refinements_{active_tool}_used"
    
    return {
        'refined_query': new_query.content,
        refinement_key: state.get(refinement_key, 0) + 1
    }

def route_after_refine(state: ResearchState):
    """
    After refining, route to the currently active tool.
    """
    if not state.get('active_tool', None):
        return "web"
    
    return state['active_tool']


def web_search_node(state: ResearchState) -> ResearchState:
    """Performs a web search."""
    
    results = search_tool.invoke({"query": state['refined_query']})
    
    if isinstance(results, str):
        # If results is a string, treat it as a single document or handle as an error
        docs = [results] if results.strip() else []
    elif isinstance(results, list):
        # If results is a list, extract 'content' from each dictionary
        docs = [res['content'] for res in results if res and isinstance(res, dict) and 'content' in res]

    return {
        "sources": docs, 
        "active_tool": "web"
    }

def arxiv_search_node(state: ResearchState) -> ResearchState:
    """Performs an ArXiv search."""
    
    search = arxiv.Search(
        query=state['refined_query'],
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    summaries = [result.summary.replace("\n", " ") for result in search.results()]
    
    return {
        "sources": summaries, 
        "active_tool": "arxiv"
    }

def rag_search_node(state: ResearchState) -> ResearchState:
    """Performs a RAG search."""
    
    retrieved_docs = retriever.invoke(state['refined_query']) or []
    
    return {
        "sources": [doc.page_content for doc in retrieved_docs],
        "active_tool": "rag"
    }


def grade_and_filter_node(state: ResearchState) -> ResearchState:
    """Grades a list of documents and returns only the relevant ones."""
    
    query = state['messages'][-1].content
    documents = state['sources']
    
    grader = llm.with_structured_output(SourceGrade)
    
    relevant_docs = []
    for doc in documents:
        prompt = f"Is the following document relevant to the user's question? Document:\n\n{doc}\n\nQuestion: {query}"
        response = grader.invoke(prompt)
        if response.related:
            relevant_docs.append(doc)
       
    # Append relevant docs to the main list
    current_related_docs = state.get('related_documents', [])
    updated_related_docs = current_related_docs + relevant_docs

    return {
        "related_documents": updated_related_docs,
        "newly_added_count": len(relevant_docs),
        "sources": [] # Clear the temporary sources list
    }

def route_after_grading(state: ResearchState):
    """
    Routes logic after the 'grade_and_filter' node.
    Decides whether to refine or advance to the next tool.
    """
    # Get the count of documents that were just added in the last step
    last_added_count = state.get("newly_added_count", 0)
    active_tool = state["active_tool"]
    
    if last_added_count > 0:
        # If docs were found, advance to the next tool in the sequence
        if active_tool == "web":
            return "arxiv"
        elif active_tool == "arxiv":
            return "rag"
        else: # After rag, we are done with searching
            return "synthesize"
    else:
        # If no docs were found, check if we can refine
        refinement_key = f"refinements_{active_tool}_used"
        if state.get(refinement_key, 0) < MAX_REFINEMENTS:
            return "need_refine"
        else:
            # If out of refinements for this tool, advance anyway to avoid getting stuck
            if active_tool == "web":
                return "arxiv"
            elif active_tool == "arxiv":
                return "rag"
            else:
                return "synthesize"
    
    
def synthesizer_node(state: ResearchState) -> ResearchState:
    """The final node that synthesizes the answer."""
    
    docs = "\n\n---\n\n".join(state.get('related_documents', []))
    
    synthesizer_messages: list[BaseMessage] = list(state["messages"])
    
    if not docs:
        no_docs_message = "After a thorough search, I could not find any relevant documents to answer your question."
        
        synthesizer_messages.append(AIMessage(content=no_docs_message))
        
        return {
            "messages": synthesizer_messages
        }

    prompt = (
        "Based ONLY on the following documents, provide a comprehensive and well-structured answer to the last user query. "
        "Do not mention the documents themselves in your answer. Synthesize the information into a single, coherent response.\n\n"
        f"Documents:\n{docs}"
    )
    
    synthesizer_messages.append(SystemMessage(content=prompt))

    response = llm.invoke(synthesizer_messages)
    
    return {
        "messages": response
    }
    