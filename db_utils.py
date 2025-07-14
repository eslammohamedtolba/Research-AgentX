import sqlite3
import uuid
from llm_utils import generate_conversation_title

DB_FILE = "db.sqlite"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def initialize_db():
    """
    Initializes the database. Renames 'created_at' to 'used_at' if the old column exists,
    then creates the 'conversations' table if it doesn't exist.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if the table exists and if the old 'created_at' column is present
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'created_at' in columns:
            # Simple migration: just rename the column for existing users
            cursor.execute("ALTER TABLE conversations RENAME COLUMN created_at TO used_at")

        # Create the table with the correct 'used_at' column if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_all_conversations():
    """
    Retrieves all conversations from the 'conversations' table, ordered by most recently used.
    """
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Order by the new 'used_at' column
        cursor.execute("SELECT * FROM conversations ORDER BY used_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def create_new_conversation():
    """
    Creates a new conversation, setting its initial 'used_at' timestamp.
    """
    thread_id = str(uuid.uuid4())
    name = "New Chat"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Ensure the 'used_at' is set on creation
        cursor.execute(
            "INSERT INTO conversations (thread_id, name, used_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (thread_id, name)
        )
        conn.commit()
    return thread_id, name

def update_conversation_timestamp(thread_id: str):
    """
    Updates the 'used_at' timestamp for a given conversation to the current time.
    This is the key function to call whenever a chat is accessed.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET used_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
            (thread_id,)
        )
        conn.commit()

def rename_conversation(thread_id: str, query: str):
    """Renames a conversation and updates its 'used_at' timestamp."""
    update_conversation_timestamp(thread_id) # Update timestamp on rename
    new_name = generate_conversation_title(query=query)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET name = ? WHERE thread_id = ?", (new_name, thread_id))
        conn.commit()
    return new_name

def delete_conversation(thread_id: str):
    """Deletes a conversation from the 'conversations' and 'checkpoints' tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        conn.commit()
        
