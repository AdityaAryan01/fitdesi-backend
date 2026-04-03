import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()


# Define paths
CHROMA_PATH = "./data/chroma_db"
DOCS_PATH = "./data/rag_docs"

def load_and_embed_pdfs():
    print("📚 Step 1: Loading PDFs using robust PyMuPDF parser...")
    
    # We use PyMuPDFLoader which handles broken/complex PDFs perfectly
    loader = DirectoryLoader(
        DOCS_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    
    raw_documents = loader.load()
    
    if not raw_documents:
        print("❌ No PDFs found in data/rag_docs/. Please add some and try again.")
        return

    print(f"✂️ Step 2: Splitting {len(raw_documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks.")

    print("🧠 Step 3: Connecting to HuggingFace (bge-large-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    print("💾 Step 4: Saving Vectors to ChromaDB...")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Successfully ingested PDFs into ChromaDB at {CHROMA_PATH}!")

if __name__ == "__main__":
    os.makedirs(DOCS_PATH, exist_ok=True)
    load_and_embed_pdfs()