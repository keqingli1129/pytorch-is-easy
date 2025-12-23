import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_url = "https://arxiv.org/pdf/2501.04040.pdf"
loader = PyPDFLoader(pdf_url)
documents = loader.load()
print(f"Loaded {len(documents)} document(s) from the PDF.")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
chunk_sizes = [len(chunk.page_content) for chunk in chunks]

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", base_url="http://localhost:11434")
idx = embeddings.embed_query("Test embedding")
print(f"Sample embedding vector (first 5 values): {idx[:5]}")

vector_store = Chroma(
    collection_name="firstchain_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_firstchain",
)

vector_store.add_documents(chunks)

print("Documents added to the vector store and persisted.")
print(f"Chunk sizes: {chunk_sizes}")
print(f"First chunk content preview: {chunks[0].page_content[:200]}...")

