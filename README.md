# Research AgentX 🤖

An autonomous, stateful research agent powered by **LangGraph**. It dynamically plans and executes research tasks, evaluates the relevance of its findings, and refines its own search queries to recover from failures. It leverages web searches, academic paper databases (ArXiv), and a local vector store to provide comprehensive, synthesized answers to complex questions.

-----

## 🖼️ Application UI

The user interface is designed for a seamless and interactive research experience. Users can ask a question and watch as the agent executes its plan, displaying the final synthesized answer.

![ResearchAgentX Application](<ResearchAgentX App.png>)

-----

## ✨ Overview

**Research AgentX** is an advanced AI agent designed to automate the research process. Given a user's query, it intelligently plans and executes a series of steps to gather information from various sources, including:

- **General Web Search**: For up-to-date information and broad topics using Tavily Search.
- **Academic Papers**: For scientific and technical knowledge from ArXiv.
- **Internal Knowledge Base**: For specialized information using a Retrieval-Augmented Generation (RAG) pipeline with a local ChromaDB instance.

The agent uses a stateful, conditional graph built with LangGraph to manage its workflow, ensuring it makes informed decisions about searching, grading, refining its queries, and synthesizing a final answer.

-----

## ⚙️ How It Works

The agent's logic is a sophisticated state machine with conditional routing that allows for intelligent recovery and planning.

1.  **Planning**: The `Research_Strategist` node (the "brain") analyzes the query and the current state to create a two-part plan: whether to refine the query (`refine: bool`) and which tool to use next (`next: tool_name`).
2.  **Conditional Routing**: The graph first checks the `refine` plan.
    - If `True`, it routes to the `refine_query` node to improve the search query.
    - If `False`, it proceeds directly to the planned tool.
3.  **Tool Execution**: The chosen search tool (`web_search`, `arxiv_search`, or `rag_search`) runs with the current query.
4.  **Relevance Grading**: After a search, the results are sent to the `grade_and_filter` node. This node uses an LLM to discard irrelevant documents, ensuring only high-quality information proceeds.
5.  **Loop & Re-plan**: Control returns to the `Research_Strategist`, which analyzes the new state (including the new documents and tool usage counts) and creates a new plan.
6.  **Synthesis**: This loop of planning, searching, and grading continues until the strategist determines the research is complete. It then calls the `synthesize` node to consolidate all relevant information into a single, coherent answer.

-----

## 🧠 Graph Architecture

The new architecture uses **conditional edges** to create a dynamic, intelligent workflow. After gathering results, a dedicated `grade_and_filter` node assesses relevance. If a search fails, the graph can now enter a `refine_query` loop to recover and retry, making the agent far more robust.

**[IMPORTANT: Replace the image below with your new graph diagram, `ResearchAgentX Graph.png`]**

![New ResearchAgentX Graph architecture](<ResearchAgentX Graph.PNG>)

-----

## 🚀 Key Features

- **Stateful Execution**: Maintains a complete state of its progress, allowing for context-aware decisions.
- **Multi-Tool Orchestration**: Dynamically chooses between web search, ArXiv, and an internal RAG pipeline.
- **Intelligent Failure Recovery**: Automatically refines and retries search queries that fail to yield relevant results.
- **Relevance-Based Grading**: Employs an LLM-powered grader to filter out irrelevant noise from search results, ensuring high-quality context for the final answer.
- **Conditional Graph Logic**: Uses LangGraph's conditional edges to dynamically route between searching, grading, refining, and synthesizing.
- **Retrieval-Augmented Generation (RAG)**: Leverages a local ChromaDB vector store for expert-level knowledge retrieval.

-----

## 🛠️ Tech Stack

- **Orchestration**: LangChain & LangGraph
- **LLM**: Google Gemini (`gemini-2.0-flash`)
- **Web Search**: Tavily Search API
- **Vector Database**: ChromaDB
- **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **Academic Search**: ArXiv Python Library
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

2.  **First-Time Setup**: The first time you run the application, it will download the ML paper dataset from Hugging Face and build the ChromaDB vector store. This may take a few minutes. Subsequent runs will be much faster.

3.  **Ask a Question**: Open your browser to the Streamlit URL, type your research question, and press Enter.

-----

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Whether it's fixing a bug, proposing a new feature, or improving documentation, your help makes this project better.