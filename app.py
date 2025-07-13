import streamlit as st
from graph import create_graph
from langchain_core.messages import HumanMessage, AIMessage
import db_utils

# Create the 'conversations' table on the first run if not exist
db_utils.initialize_db()

# Streamlit setup
st.set_page_config(page_title="Research AgentX", page_icon="🤖")
st.title("Research AgentX 🤖")
st.caption("A stateful research agent powered by LangGraph and SQLite.")

# The graph is created once and stored in session state
if 'agent_app' not in st.session_state:
    st.session_state.agent_app = create_graph()

app = st.session_state.agent_app

# This logic now handles creating new chats and loading existing ones.
if "thread_id" not in st.session_state:
    # Check if there are any conversations left to fall back to
    all_convs = db_utils.get_all_conversations()
    if all_convs:
        st.session_state.thread_id = all_convs[0]['thread_id']
        st.session_state.thread_name = all_convs[0]['name']
    else:
        # If no conversations exist, create a new one
        thread_id, thread_name = db_utils.create_new_conversation()
        st.session_state.thread_id = thread_id
        st.session_state.thread_name = thread_name

# The config is now created dynamically based on the session's thread_id
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "recursion_limit": 50
    }
}

with st.sidebar:
    st.header("Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        thread_id, thread_name = db_utils.create_new_conversation()
        st.session_state.thread_id = thread_id
        st.session_state.thread_name = thread_name
        st.rerun()

    st.divider()

    conversations = db_utils.get_all_conversations()
    for conv in conversations:
        # Create a single row with two columns: name (left), delete (right)
        cols = st.columns([0.85, 0.15])
        
        with cols[0]:
            button_type = "primary" if conv['thread_id'] == st.session_state.thread_id else "secondary"
            if st.button(conv['name'], key=f"select_{conv['thread_id']}", use_container_width=True, type=button_type):
                st.session_state.thread_id = conv['thread_id']
                st.session_state.thread_name = conv['name']
                st.rerun()
        
        with cols[1]:
            if st.button("X", key=f"delete_{conv['thread_id']}", help="Delete this conversation"):
                db_utils.delete_conversation(conv['thread_id'])
                if conv['thread_id'] == st.session_state.thread_id:
                    del st.session_state.thread_id
                st.rerun()

# Display full history (user and assistant pairs)
def display_chat_history():
    """Displays messages from the history."""
    
    # Get the current state to access messages
    history = app.get_state(config)
    if not history:
        return
    
    # Get the message list from the latest state
    for msg in history.values.get('messages', []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

# Handle streaming + display steps inline
def stream_response(initial_state: str):
    """Streams the agent's response and displays status updates for each step."""
    
    with st.status("Thinking...", expanded=True) as status:
        final_answer = None
        # Stream the response using the new initial_state
        for step in app.stream(initial_state, config=config):
            
            node = list(step.keys())[0]
            
            if node == "web_search":
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
                
                final_answer = output["messages"].content
                status.update(label="✅ Done!", state="complete")
                
        yield final_answer or "The agent finished without providing a final answer."

# Main chat UI
display_chat_history()

# Chat input
if prompt := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # If this was the first message in the chat, trigger the rename.
    if not app.get_state(config):
        db_utils.rename_conversation(st.session_state.thread_id, prompt)

    with st.chat_message("assistant"):
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "refined_query": prompt,
            "sources": [],
            "related_documents": [],
            "refinements_web_used": 0,
            "refinements_arxiv_used": 0,
            "refinements_rag_used": 0,
            "active_tool": "web"
        }
        st.write_stream(stream_response(initial_state))

    # Rerun to clear the input box and potentially update history display
    st.rerun()


