import streamlit as st
from graph import create_graph
from langchain_core.messages import HumanMessage, AIMessage
import db_utils
import time

# Create the 'conversations' table on the first run if it doesn't exist
db_utils.initialize_db()

# Streamlit Page Setup 
st.set_page_config(page_title="Research AgentX", page_icon="🤖", layout="wide")
st.title("Research AgentX 🤖")
st.caption("A stateful research agent powered by LangGraph and SQLite.")

# Agent and Session State Initialization
if 'agent_app' not in st.session_state:
    st.session_state.agent_app = create_graph()

# Initialize thread_id if it's not already set
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Sidebar for Conversation Management
with st.sidebar:
    st.header("Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        thread_id, thread_name = db_utils.create_new_conversation()
        st.session_state.thread_id = thread_id
        st.rerun()

    st.divider()

    conversations = db_utils.get_all_conversations()
    for conv in conversations:
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            is_selected = (st.session_state.thread_id and conv['thread_id'] == st.session_state.thread_id)
            button_type = "primary" if is_selected else "secondary"
            if st.button(conv['name'], key=f"select_{conv['thread_id']}", use_container_width=True, type=button_type):
                st.session_state.thread_id = conv['thread_id']
                db_utils.update_conversation_timestamp(conv['thread_id'])
                st.rerun()
        
        with cols[1]:
            if st.button("🗑️", key=f"delete_{conv['thread_id']}", help="Delete this conversation"):
                db_utils.delete_conversation(conv['thread_id'])
                if conv['thread_id'] == st.session_state.thread_id:
                    st.session_state.thread_id = None
                st.rerun()

# Main Chat Interface
def display_chat_history(app, config):
    """Displays messages from the history for the given config."""
    history = app.get_state(config)
    if not history:
        return
    
    messages = history.values.get('messages', [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

def stream_response(app, initial_state, config):
    """Streams the agent's response."""
    with st.status("Thinking...", expanded=True) as status:
        final_answer = None
        for step in app.stream(initial_state, config=config):
            node = list(step.keys())[0]
            if node == "web_search":
                status.update(label="🔍 Searching the Web...")
            elif node == "arxiv_search":
                status.update(label="📄 Searching ArXiv...")
            elif node == "rag_search":
                status.update(label="📚 Searching Knowledge Base...")
            elif node == "grade_and_filter":
                status.update(label="⚖️ Grading and filtering...")
            elif node == "refine_query":
                status.update(label="✍️ Refining query...")
            elif node == "synthesize":
                output = step[node]
                if "messages" in output and output["messages"]:
                    final_answer = output["messages"].content
                status.update(label="✅ Done!", state="complete")
            time.sleep(0.5)
        yield final_answer or "The agent finished without providing a final answer."


# Display the history of the currently selected chat, if any.
if st.session_state.thread_id:
    app = st.session_state.agent_app
    config = {"configurable": {"thread_id": st.session_state.thread_id, "recursion_limit": 50}}
    display_chat_history(app, config)

# Always display the chat input box at the bottom.
if prompt := st.chat_input("Ask a research question..."):
    
    # If no chat is selected, create a new one automatically.
    if not st.session_state.thread_id:
        thread_id, thread_name = db_utils.create_new_conversation()
        st.session_state.thread_id = thread_id
    
    # Create the config for this run.
    app = st.session_state.agent_app
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "recursion_limit": 50
        }
    }

    # Update the timestamp because the conversation is being used.
    db_utils.update_conversation_timestamp(st.session_state.thread_id)

    # Display the user's message.
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if this is the very first message to trigger the rename.
    history = app.get_state(config)
    if not history.values.get("messages"):
        db_utils.rename_conversation(st.session_state.thread_id, prompt)
    
    # Stream the assistant's response.
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
        st.write_stream(stream_response(app, initial_state, config))
    
    # Rerun to update the sidebar with the new name and order.
    st.rerun()
