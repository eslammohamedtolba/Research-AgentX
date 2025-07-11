import streamlit as st
from graph import create_graph
import uuid
import time

# Streamlit setup
st.set_page_config(page_title="Thoth Chat 🤖", page_icon="🤖")
st.title("Research AgentX 🤖")
st.caption("A stateful research agent powered by LangGraph.")

# Agent and thread setup
if 'agent_app' not in st.session_state:
    st.session_state.agent_app = create_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

app = st.session_state.agent_app
config = {"configurable": {"thread_id": st.session_state.thread_id, "recursion_limit": 10}}

# Display full history (user ↔ assistant pairs)
def display_chat_history():
    """Displays the chat history from the agent's memory without repetition."""
    history = list(app.get_state_history(config))
            
    displayed_queries = set()
    displayed_answers = set()

    for state in reversed(history):
        query = state.values.get("query")
        final_answer = state.values.get("final_answer")

        # Avoid showing duplicate queries
        if query in displayed_queries or final_answer in displayed_answers:
            continue
        
        displayed_queries.add(query)
        displayed_answers.add(final_answer)

        if query:
            with st.chat_message("user"):
                st.markdown(query)

        if final_answer:
            with st.chat_message("assistant"):
                st.markdown(final_answer)

# Handle streaming + display steps inline
def stream_response(query: str):
    with st.status("Thinking...", expanded=True) as status:
        final_answer = None
        for step in app.stream({"query": query}, config=config):
            node = list(step.keys())[0]
            if node == "Research_Strategist_node":
                status.update(label="🧠 Strategizing...")
            elif node == "web_search_node":
                status.update(label="🔍 Searching the Web...")
            elif node == "fetch_arxiv_node":
                status.update(label="📄 Searching ArXiv...")
            elif node == "rag_search_node":
                status.update(label="📚 Searching the knowledge base...")
            elif node == "Synthesizer_node":
                output = step[node]
                final_answer = output.get("final_answer", "No answer.")
                status.update(label="✅ Done!", state="complete")
            time.sleep(1)
        yield final_answer or "No final answer found."

# --- Main chat UI ---
display_chat_history()

# Chat input
if prompt := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.write_stream(stream_response(prompt))

    st.rerun()

