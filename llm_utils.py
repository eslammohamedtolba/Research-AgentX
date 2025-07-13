from config import llm

def generate_conversation_title(query: str) -> str:
    """
    Uses an LLM to generate a short, descriptive title for a conversation.
    """
    prompt = (
        "You are an expert at summarizing conversations. Your task is to create a concise, "
        "descriptive title (3-4 words maximum) for a new chat session based on the user's first query. "
        "The title should capture the main topic of the query.\n\n"
        f"User's first query: \"{query}\"\n\n"
        "Return ONLY the title itself, with no extra text or quotation marks."
    )
    
    try:
        response = llm.invoke(prompt)
        # Clean up the response to remove potential quotes or extra whitespace
        title = response.content.strip().strip('"')
        return title if title else "New Chat"
    except Exception as e:
        return "New Chat"

