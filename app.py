import streamlit as st
from graph import create_graph
import uuid
import time

# Streamlit setup
st.set_page_config(page_title="Research AgentX 🤖", page_icon="🤖")
st.title("Research AgentX 🤖")
st.caption("A stateful research agent powered by LangGraph.")

# Agent and thread setup
if 'agent_app' not in st.session_state:
    st.session_state.agent_app = create_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

app = st.session_state.agent_app
config = {"configurable": {"thread_id": st.session_state.thread_id, "recursion_limit": 50}}

# Display full history (user ↔ assistant pairs)
def display_chat_history():
    """Displays the final user/assistant messages from the history."""
    
    messages_to_display = []
    processed_queries = set()

    history = app.get_state_history(config)
    
    # Iterate backwards through history to find the latest state for each query
    for state in reversed(list(history)):
        query = state.values.get("query")
        final_answer = state.values.get("final_answer")

        # Find the first completed Q&A pair for a query and add it
        if query and query not in processed_queries and final_answer:
            messages_to_display.extend([
                {"role": "assistant", "content": final_answer},
                {"role": "user", "content": query}
            ])
            processed_queries.add(query)

    # Display messages in chronological order (oldest at the top)
    for msg in reversed(messages_to_display):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle streaming + display steps inline
def stream_response(initial_state: str):
    """Streams the agent's response and displays status updates for each step."""
    
    with st.status("Thinking...", expanded=True) as status:
        final_answer = None
        # Stream the response using the new initial_state
        for step in app.stream(initial_state, config=config):
            
            node = list(step.keys())[0]
            
            if node == "Research_Strategist":
                status.update(label="🧠 Strategizing the next step...")
            elif node == "web_search":
                status.update(label="🔍 Searching the Web...")
            elif node == "arxiv_search":
                status.update(label="📄 Searching ArXiv for papers...")
            elif node == "rag_search":
                status.update(label="📚 Searching the RAG knowledge base...")
            elif node == "grade_and_filter":
                status.update(label="⚖️ Grading and filtering documents...")
            elif node == "refine_query":
                status.update(label="✍️ Refining the search query...")
            elif node == "synthesize":
                output = step[node]
                final_answer = output.get("final_answer", "No answer could be generated.")
                status.update(label="✅ Done!", state="complete")
            time.sleep(5)
        yield final_answer or "The agent finished without providing a final answer."

# --- Main chat UI ---
display_chat_history()

# Chat input
if prompt := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        initial_state = {
            "query": prompt,
            "refined_query": prompt,
            "sources": [],
            "related_documents": [],
            "refinements_used": 0,
            "web_count": None,
            "arxiv_count": None,
            "rag_count": None,
            "final_answer": "",
        }
        st.write_stream(stream_response(initial_state))

    # Rerun to clear the input box and potentially update history display
    st.rerun()

