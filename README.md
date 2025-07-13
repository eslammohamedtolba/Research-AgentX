# Research AgentX 🤖

A conversational, multi-turn, and stateful research agent powered by **LangGraph**. It follows a deterministic workflow to search multiple data sources, intelligently refines its queries based on conversational context, and synthesizes its findings into a coherent answer. All conversation history is persistent, powered by a SQLite database backend.

-----

## 🖼️ Application UI

The user interface supports multiple, persistent conversations. Users can switch between chats, create new ones, and see the agent work through its research process in real-time.

![ResearchAgentX Application](<ResearchAgentX App.png>)

-----

## ✨ Overview

**Research AgentX** has evolved into a sophisticated conversational research assistant. It maintains the context of a conversation across multiple turns, allowing for follow-up questions and iterative exploration of topics.

Given a user's query, it executes a structured, deterministic plan to gather information from:

- **General Web Search**: For up-to-date information using Tavily Search.
- **Academic Papers**: For scientific knowledge from ArXiv.
- **Internal Knowledge Base**: For specialized information using a Retrieval-Augmented Generation (RAG) pipeline with a local ChromaDB instance.

The agent uses a stateful graph built with LangGraph and a **SQLite checkpointer** to manage its workflow and persist all conversation history, ensuring no data is lost between sessions.

-----

## ⚙️ How It Works

The agent's logic has been refactored from an LLM-based planner to a more robust and predictable deterministic workflow.

1.  **Query Refinement**: Every user query first enters the `refine_query` node. This node uses an LLM with a highly detailed prompt to analyze the full conversation history. It rewrites the user's latest message into a complete, standalone search query, resolving context and handling follow-up questions.
2.  **Sequential Search & Grading**: The agent proceeds through its tools in a fixed order: `web_search`, then `arxiv_search`, then `rag_search`.
    - After each search, the results are passed to the `grade_and_filter` node, which uses an LLM to discard irrelevant documents.
3.  **Conditional Routing**: The `route_after_grading` function checks if the last search found any relevant documents.
    - If **YES**, it proceeds to the next tool in the sequence (e.g., from `web_search` to `arxiv_search`).
    - If **NO**, it loops back to the `refine_query` node to try a different query for the *same* tool. It will retry up to a defined limit before moving on.
4.  **Synthesis**: Once all search tools have been tried, the graph proceeds to the `synthesize` node. This node takes all the relevant documents collected from all sources and generates a final, comprehensive answer based on the user's last question.
5.  **Persistence**: The entire state of the conversation, including every message and intermediate step, is automatically saved to a `db.sqlite` file using LangGraph's `SqliteSaver`.


-----

## 🧠 Graph Architecture

The new architecture is a deterministic, sequential graph that prioritizes reliability and predictability. It uses conditional edges to loop for query refinement when a search tool fails to find useful information. This design removes the "LLM-in-the-loop" for planning, resulting in a more stable and debuggable agent.

![New ResearchAgentX Graph architecture](<ResearchAgentX Graph.PNG>)

-----

## 🚀 Key Features

- **Conversational Memory**: Maintains the full context of a conversation, allowing for natural follow-up questions.
- **Persistent Chat History**: Uses a SQLite database to save all conversations, so you never lose your work.
- **Multi-Conversation Management**: A sidebar UI allows you to create, delete, and switch between different chat sessions.
- **Automatic Conversation Titling**: Automatically generates a concise title for new chats based on your first query.
- **Intelligent Query Refinement**: A dedicated node analyzes user intent and conversational history to create highly effective search queries.
- **Multi-Tool Orchestration**: Sequentially queries the web, ArXiv, and an internal RAG pipeline.
- **Relevance-Based Grading**: Employs an LLM-powered grader to filter out irrelevant noise from search results.


-----

## 🛠️ Tech Stack

- **Orchestration**: LangChain & LangGraph
- **LLM**: Google Gemini (`gemini-2.0-flash`)
- **Database**: SQLite (for conversation state) & ChromaDB (for RAG)
- **Web Search**: Tavily Search API
- **Academic Search**: ArXiv Python Library
- **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **UI Framework**: Streamlit

-----

## 📦 Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/eslammohamedtolba/Research-AgentX.git
    cd Research-AgentX
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows PowerShell
    .\venv\Scripts\Activate.ps1
    
    # On macOS/Linux, use: source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    - Create a file named `.env` in the root directory.
    - Add your API keys to the `.env` file:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    ```

-----

## ▶️ How to Run

1.  **Launch the Application**
    ```bash
    streamlit run app.py
    ```

2.  **First-Time Setup**: The first time you run the application, it will download the ML paper dataset from Hugging Face and build the ChromaDB vector store. This may take a few minutes. Subsequent runs will be much faster as it will load the existing database.

3.  **Start Chatting**: Open your browser to the Streamlit URL. A new chat will be created automatically. Type your research question and press Enter.

-----

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Whether it's fixing a bug, proposing a new feature, or improving documentation, your help makes this project better.