from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model="qwen3:0.6b", 
    base_url="http://localhost:11434", 
    temprature = 0.1)

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b", 
    base_url="http://localhost:11434")

# idx = embeddings.embed_query("Test embedding")
# print(idx)
vector_store = Chroma(
    collection_name="firstchain_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_firstchain",
)
print("Vector store initialized.")
# vectorstore = Chroma(persist_directory="./chroma_db")
# query = "What is the main methods available for RAG?" 
# results = vector_store.similarity_search(query, k=3)
# print(f"Top {len(results)} results for the query '{query}':")
# for i, res in enumerate(results):
#     print(f"Result {i+1} preview: {res.page_content[:200]}...")

answer =llm.invoke("Hello")
print(answer)