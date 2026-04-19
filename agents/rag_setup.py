import os
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables (e.g., GOOGLE_API_KEY) right away
load_dotenv()

def get_retriever():
    """
    Initializes the Chroma vector store and returns a retriever configured
    to fetch the top 2 most relevant chunks based on semantic search.
    """
    
    db_dir = "./chroma_db"
    doc_path = "data/retention_best_practices.md"
    
    # 1. Initialize local embedding model to save API limits
    # all-MiniLM-L6-v2 is fast and effective for short paragraphs/sentences
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if the vector DB has already been populated
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        print(f"Building vector database from {doc_path}...")
        
        # 2. Load the knowledge base document
        try:
            loader = TextLoader(doc_path, encoding='utf-8')
            documents = loader.load()
        except Exception as e:
            raise FileNotFoundError(f"Knowledge base not found at {doc_path}. Run data setup first.") from e
        
        # 3. Chunk the document appropriately
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, # Optimized chunk_size as requested
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # 4. Load chunks into the local Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=db_dir
        )
        print("Successfully created and persisted Chroma database.")
    else:
        print("Loading existing Chroma vector database from disk...")
        vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings
        )
        
    # 5. Return retriever configured for top k results
    return vectorstore.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    print("=== Initializing RAG Setup ===")
    retriever = get_retriever()
    
    if retriever:
        # Test the retrieval pipeline
        test_query = "Customer is leaving because their monthly bill is too high"
        print(f"\n[Test Query]: '{test_query}'\n")
        
        results = retriever.invoke(test_query)
        
        print("=== Retrieved Results ===")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print("-" * 40)
