import sqlite3
import uuid
from llm_utils import generate_conversation_title

DB_FILE = "db.sqlite"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # check_same_thread=False is needed for Streamlit's multi-threading
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def initialize_db():
    """
    Initializes the database by creating a 'conversations' table if it doesn't exist.
    This table will store a user-friendly name for each thread_id.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_all_conversations():
    """
    Retrieves all conversations from the 'conversations' table, ordered by creation time.
    """
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row # This allows accessing columns by name
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def create_new_conversation():
    """
    Creates a new conversation with a unique thread_id and a default name.
    Returns the thread_id and the name of the newly created conversation.
    """
    thread_id = str(uuid.uuid4())
    # Generate a default name
    name = "New Chat"
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (thread_id, name) VALUES (?, ?)",
            (thread_id, name)
        )
        conn.commit()
    return thread_id, name

def rename_conversation(thread_id: str, query: str):
    """Renames a conversation in the database."""
    
    # Generate the new name
    new_name = generate_conversation_title(query=query)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET name = ? WHERE thread_id = ?", (new_name, thread_id))
        conn.commit()
        
def delete_conversation(thread_id: str):
    """Deletes a conversation from the 'conversations' table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # We also need to delete the checkpoints from LangGraph's table
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        conn.commit()