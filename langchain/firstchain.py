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

query = "What is the main methods available for RAG?" 
results = vector_store.similarity_search(query, k=3)
print(f"Top {len(results)} results for the query '{query}':")
for i, res in enumerate(results):
    print(f"Result {i+1} preview: {res.page_content[:200]}...")
print("Vector store setup and query completed.")

similarity_retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 2}
)

# similar_docs = similarity_retriever.get_relevant_documents("Explain the concept of RAG.")
# print(f"Retrieved {len(similar_docs)} documents using the retriever.")
# for i, doc in enumerate(similar_docs):
#     print(f"Retrieved Document {i+1} preview: {doc.page_content[:200]}...")
# print("Retriever test completed.")

if chunks:
    sample_metadata = chunks[0].metadata
    print(f"Sample metadata: {sample_metadata}")
    
    # Get unique page numbers for filtering examples
    page_numbers = set()
    for chunk in chunks[:10]:  # Check first 10 chunks
        if 'page' in chunk.metadata:
            page_numbers.add(chunk.metadata['page'])
    print(f"Available page numbers (sample): {sorted(list(page_numbers))[:5]}...")

if page_numbers:
    target_page = sorted(list(page_numbers))[0]  # Use first available page
    page_results = vector_store.similarity_search(
        "methodology approach",
        k=10,
        filter={"page": target_page}
    )
    print(f"Searching only in Page {target_page}:")
    for i, doc in enumerate(page_results, 1):
        print(f"  Result {i}: Page {doc.metadata.get('page')} - {doc.page_content[:150]}...")

complex_results = vector_store.similarity_search(
    "research findings",
    k=2,
    filter={
        "$and": [
            {"page": {"$lte": 10}},  # Page 0 or higher
            {"source": {"$ne": ""}}  # Has a source
        ]
    }
)

print("Using complex filter (page >= 0 AND has source):")
for i, doc in enumerate(complex_results, 1):
    print(f"  Result {i}: Page {doc.metadata.get('page')} - {doc.page_content[:150]}...")

final_query = "What are the main LLM models used for RAG?"
context_docs = similarity_retriever.invoke(final_query)

print(f"\nQuery: '{final_query}'")
print(f"✓ Retrieved {len(context_docs)} relevant document chunks")

for i, doc in enumerate(context_docs[:2], 1):  # Show first 2 for brevity
    print(f"\nChunk {i}: {doc.page_content[:250]}...")
    