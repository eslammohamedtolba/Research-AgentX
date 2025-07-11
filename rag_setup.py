import os
from langchain_core.documents import Document
from datasets import load_dataset
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"

def get_retriever(embedding_function):
    """Creates or loads the ChromaDB and returns a retriever."""
    if not os.path.exists(CHROMA_PATH):
        print("Database not found. Creating and populating a new one... 📚")
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        docs = [
            Document(
                page_content=row["abstract"],
                metadata={"title": row["title"]}
            ) for row in dataset.select(range(20000))
        ]
        db = Chroma.from_documents(docs, embedding_function, persist_directory=CHROMA_PATH)
        print("Database created successfully.")
    else:
        print("Existing database found. Loading... ⚡️")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    return db.as_retriever()