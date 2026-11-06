import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Import your database and the new RagDocument table
from database import SessionLocal
from models import RagDocument

load_dotenv()

DOCS_PATH = "./data/rag_docs"

def load_and_embed_pdfs():
    print("📚 Step 1: Loading PDFs using PyMuPDF parser...")
    
    loader = DirectoryLoader(
        DOCS_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    
    raw_documents = loader.load()
    
    if not raw_documents:
        print("❌ No PDFs found in data/rag_docs/. Please add some and try again.")
        return

    # Metadata Hack (So the LLM knows which PDF/Page it's reading)
    for doc in raw_documents:
        filename = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
        page_num = doc.metadata.get('page', 0) + 1
        doc.page_content = f"[SOURCE: {filename} | PAGE: {page_num}]\n" + doc.page_content

    print(f"✂️ Step 2: Splitting {len(raw_documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=400,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks.")

    print("🧠 Step 3: Loading HuggingFace Embedding Model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Extract just the text strings to feed into the embedder
    texts = [chunk.page_content for chunk in chunks]
    
    print("⏳ Generating mathematical vectors (this might take a moment)...")
    embedded_vectors = embeddings_model.embed_documents(texts)

    print("💾 Step 4: Saving directly to PostgreSQL (pgvector)...")
    db = SessionLocal()
    try:
        # Clear out old documents so we don't get duplicates if you run this twice
        db.query(RagDocument).delete()
        
        # Loop through our chunks and their matching vectors, and save them to SQL
        for i, chunk in enumerate(chunks):
            new_doc = RagDocument(
                content=chunk.page_content,
                embedding=embedded_vectors[i], # The 1024-dimension math array
                metadata_text=str(chunk.metadata)
            )
            db.add(new_doc)
            
        db.commit()
        print(f"✅ SUCCESS! {len(chunks)} PDF chunks securely saved to PostgreSQL!")
    except Exception as e:
        print(f"❌ Database error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    os.makedirs(DOCS_PATH, exist_ok=True)
    load_and_embed_pdfs()