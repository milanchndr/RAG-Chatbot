import fitz # PyMuPDF
import numpy as np
import google.generativeai as genai
import faiss
import pickle
import os
from dotenv import load_dotenv
load_dotenv()


# Initialize Gemini API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = genai


def extract_text(pdf_path):
    """Extract text using PyMuPDF"""
    text_chunks = []

    doc = fitz.open(pdf_path)

    for page in doc:
        # Extract text blocks as chunks
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if text:
                # Split long blocks into smaller chunks using double newlines or lines
                sub_chunks = text.split('\n\n')
                for chunk in sub_chunks:
                    lines = [line.strip() for line in chunk.split('\n') if line.strip()]
                    if lines:
                        text_chunks.append('\n'.join(lines))

    return text_chunks


def embed_chunks(text_chunks, title="finance"):

    EMBEDDING_MODEL_ID = "embedding-001"

    try:
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL_ID}",
            content=text_chunks,
            task_type="retrieval_document",
            title=title
        )
        embeddings = result['embedding']

        print(f"Successfully embedded {len(text_chunks)} text chunks.")
        print(f"Generated {len(embeddings)} embedding vectors.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return embeddings

def create_faiss_index(embeddings,text_chunks):
    """Store embedding vectors in a FAISS index."""
    if not embeddings:
        print("No embeddings")
        return None

    embeddings_np = np.array(embeddings).astype('float32')
    d = embeddings_np.shape[1]

    print(f"Creating a FAISS index with dimensionality {d}.")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    faiss.write_index(index, "faiss_index.index")
    
    # Save metadata for text chunks
    with open("metadata.pkl", "wb") as f:
        pickle.dump(text_chunks, f)
    
    print(f"Successfully added {index.ntotal} vectors to the FAISS index.")
    return index

if __name__ == "__main__":
    print("Starting ingestion process...")
    
    all_text_chunks = []
    
    # Read the list of sources 
    print("Reading data sources from config...")
    for source in config.DATA_SOURCES:
        print(f"Processing source: {source}")
        chunks = load_from_source(source)
        if chunks:
            all_text_chunks.extend(chunks)
    
    print(f"\n--- Ingestion Complete: Total {len(all_text_chunks)} chunks collected ---")

    if all_text_chunks:
        
        embeddings = embed_chunks(all_text_chunks, title=config.DOCUMENT_TITLE)
        
        if embeddings:
            create_faiss_index(embeddings, all_text_chunks)
            print("\nProcessing complete. Unified index has been created.")