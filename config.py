from langchain_tavily import TavilySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize models and tools once and import them elsewhere
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
search_tool = TavilySearch(max_results=3)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

