# Research AgentX 🤖

an autonomous, stateful research agent powered by **LangGraph**. It dynamically plans and executes research tasks by leveraging web searches, academic paper databases (ArXiv), and a local vector store to provide comprehensive, synthesized answers to complex questions.

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

The agent uses a stateful graph built with LangGraph to manage its workflow, ensuring it remembers previous results and makes informed decisions about what to do next.

-----

## ⚙️ How It Works

The agent's logic is built around a cyclical graph structure where a central "strategist" node decides the best course of action at each step.

1.  **Query Input**: The user provides a research question.
2.  **Strategist**: The `Research_Strategist_node` (the "brain") analyzes the query and the current state (what information has been gathered so far). It then decides which tool to use next.
3.  **Tool Execution**: The agent calls one of its tools (`web_search_node`, `fetch_arxiv_node`, or `rag_search_node`) to gather data.
4.  **State Update**: The results from the tool are added to the agent's state.
5.  **Loop**: Control returns to the Strategist, which re-evaluates the state and decides the next step. This loop continues until the strategist determines it has enough information.
6.  **Synthesis**: The `Synthesizer_node` is called to consolidate all the gathered information into a single, coherent, and well-structured answer.

-----

## 🧠 Graph Architecture

The core of the agent is a `StateGraph` from LangGraph. The graph defines the possible paths of execution, with the `Research_Strategist_node` acting as the central router. This architecture allows the agent to dynamically loop through different tools until a satisfactory amount of information is collected for the final synthesis.

![ResearchAgentX Graph architecture](<ResearchAgentX Graph.PNG>)

-----

## 🚀 Key Features

  - **Stateful Execution**: The agent maintains a complete state of its progress, allowing it to make smarter, context-aware decisions.
  - **Multi-Tool Orchestration**: Dynamically chooses between web search, ArXiv, and an internal RAG pipeline for comprehensive data collection.
  - **Retrieval-Augmented Generation (RAG)**: Leverages a local ChromaDB vector store populated with thousands of machine learning papers for expert-level knowledge retrieval.
  - **Structured Decision Making**: Uses a Pydantic model to force the strategist LLM into making a clean, predictable decision at each step.
  - **Modular & Extensible**: The graph-based architecture makes it easy to add new tools, modify the research logic, or swap out models.

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

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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

2.  **First-Time Setup**: The first time you run the application, it will download the ML paper dataset from Hugging Face and build the ChromaDB vector store. This process may take a few minutes. Subsequent runs will be much faster as they will load the existing database.

3.  **Ask a Question**: Open your browser to the Streamlit URL, type your research question in the input box, and press Enter.

-----

## 🤝 Contributing

Contributions are welcome and greatly appreciated\! Whether it's fixing a bug, proposing a new feature, or improving documentation, your help makes this project better.
