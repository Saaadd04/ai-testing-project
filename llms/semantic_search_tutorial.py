from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
# load_dotenv()
#
# def initialize_embeddings(api_key):
#     """Initialize Mistral embeddings, fall back to HuggingFace if it fails."""
#     try:
#         return MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
#     except Exception as e:
#         print(f"MistralAIEmbeddings failed: {e}. Using HuggingFaceEmbeddings.")
#         return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# if __name__ == "__main__":
#     # Get API key
#     api_key = os.getenv("MISTRAL_API_KEY")
#     if not api_key:
#         print("Error: MISTRAL_API_KEY not found in .env")
#         exit(1)
#     if not os.getenv("HF_TOKEN"):
#         print("Error: HF_TOKEN not found in .env")
#         exit(1)
#
#     # Load PDF
#     file_path = "C:/Users/SAAD/Downloads/Nike10k2021.pdf"
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()
#     print(f"Loaded PDF with {len(docs)} pages")
#
#     # Split into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     all_splits = text_splitter.split_documents(docs)
#     print(f"Total splits created: {len(all_splits)}")
#
#     # Initialize embeddings
#     # embeddings = initialize_embeddings(api_key)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#     # Store embeddings
#     vector_store = InMemoryVectorStore(embeddings)
#     vector_store.add_documents(documents=all_splits)
#     print(f"Stored {len(all_splits)} document chunks")
#
#     # Query 1: Nike headquarters
#     query_1 = "Where is the headquarters of Nike Inc.?"
#     results = vector_store.similarity_search(query_1)
#     print(f"\nResults for '{query_1}':")
#     print(results[0].page_content if results else "No results found")
#
#     # Query 2: Nike factory stores
#     query_2 = "What are the total number of NIKE brand factory stores in the U.S?"
#     results = vector_store.similarity_search(query_2)
#     print(f"\nResults for '{query_2}':")
#     print(results[0].page_content if results else "No results found")




os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

if __name__ == "__main__":
    # Load PDF
    file_path = "C:/Users/SAAD/Downloads/Nike10k2021.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded PDF with {len(docs)} pages")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased for better context
        chunk_overlap=400  # Increased for continuity
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Total splits created: {len(all_splits)}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store embeddings
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    print(f"Stored {len(all_splits)} document chunks")

    # Query 1: Nike headquarters
    query_1 = "Where is the headquarters of Nike Inc.?"
    results = vector_store.similarity_search(query_1, k=1)
    print(f"\nResults for '{query_1}':")
    print(results[0].page_content if results else "No results found")

    # Query 2: Nike factory stores
    query_2 = "How many NIKE brand factory stores are in the United States?"
    results = vector_store.similarity_search(query_2, k=3)  # Top 3 for better chance
    print(f"\nResults for '{query_2}':")
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res.page_content[:500]}...")