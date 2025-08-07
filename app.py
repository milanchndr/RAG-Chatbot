import faiss
import pickle
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request

import os
from dotenv import load_dotenv

import config
from data_loader import load_from_source
from create_embed_doc import extract_text, embed_chunks, create_faiss_index

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# GLOBAL VARIABLES 
faiss_index = None
text_chunks = None

def initialize_rag_system():
    """Checks for index files and creates not found."""
    global faiss_index, text_chunks
    
    # Use file paths
    if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE):
        print(f"Loading existing FAISS index from '{config.FAISS_INDEX_FILE}'...")
        faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
        with open(config.METADATA_FILE, 'rb') as f:
            text_chunks = pickle.load(f)
        print("Load complete.")
    else:
        print("Index not found. Building new index")
        
        all_chunks = []
        # Load data from all sources
        for source in config.DATA_SOURCES:
            print(f"--> Loading from source: {source}")
            chunks_from_source = load_from_source(source)
            if chunks_from_source:
                all_chunks.extend(chunks_from_source)
        
        if not all_chunks:
            raise RuntimeError("Failed to load any data")
        
        # Use document title
        embeddings = embed_chunks(all_chunks, title=config.DOCUMENT_TITLE)
        if not embeddings:
            raise RuntimeError("Failed to create embeddings.")
        
        index = create_faiss_index(embeddings, all_chunks)
        if not index:
            raise RuntimeError("Failed to create FAISS index.")
            
        faiss_index = index
        text_chunks = all_chunks
        print("\n --- Index created successfully! ---\n")
        
        

def retrieve_contextual_chunks(query, top_k=11, window_size=4):
    """Retrieve contextual chunks using FAISS index"""
    global faiss_index, text_chunks
    
    if faiss_index is None or text_chunks is None:
        raise ValueError("FAISS index or text chunks not loaded")
    
    try:
        # Generate query embedding
        result = genai.embed_content(
            model="models/embedding-001", 
            content=query, 
            task_type="retrieval_query"
        )
        query_embedding = np.array([result['embedding']]).astype('float32')
        
        # Find the most relevant chunks
        distances, indices = faiss_index.search(query_embedding, top_k)
        
        # Get the center index (most relevant chunk)
        center_index = indices[0][0]
        print(f"Most relevant chunk is at index: {center_index}")
        
        # Calculate context window
        start_index = max(0, center_index - window_size)
        end_index = min(len(text_chunks) - 1, center_index + window_size)
        
        # Get the chunks within the window
        contextual_indices = list(range(start_index, end_index + 1))
        retrieved_chunks = [text_chunks[i] for i in contextual_indices]
        
        print(f"Returning context window from index {start_index} to {end_index}")
        return retrieved_chunks
        
    except Exception as e:
        print(f"Error in retrieve_contextual_chunks: {e}")
        return []

def generate_final_answer(query, retrieved_chunks):
    """Generate final answer using retrieved chunks"""
    context = "\n".join(retrieved_chunks)

    # Prompt template
    prompt_template = f"""
    DOCUMENT CONTEXT:
    {context}

    USER QUESTION: {query}

    INSTRUCTIONS:
    1. Analyze the provided context carefully to find relevant information
    2. Answer the question using ONLY the information from the context above
    3. Structure your response clearly with proper formatting
    4. If you need to reference specific parts, mention them naturally
    5. If the context doesn't contain sufficient information to answer the question, clearly state this limitation

    RESPONSE FORMATTING GUIDELINES:
    - Use <strong>text</strong> for important terms or concepts
    - Use <ul><li>item</li></ul> for bullet points
    - Use <ol><li>item</li></ol> for numbered lists
    - Use <p>text</p> for paragraphs
    - Use <br> for line breaks when needed
    - Write in a clear, conversational tone
    - Keep your answer focused and relevant to the question
    - If the answer requires multiple steps or parts, organize them logically

    IMPORTANT: Your entire response should be properly formatted HTML that will display well in a web browser.

    Please provide your answer now:"""

    # Initialize the generative model
    model = genai.GenerativeModel('gemini-2.5-flash')
    # Generate the final answer
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        print(f"Error in generate_final_answer: {e}")
        return f"An error occurred during answer generation: {e}"

@app.route("/", methods=["GET", "POST"])
def home():
    final_answer = ""
    original_query = ""
    
    if request.method == "POST":
        query = request.form.get("query")
        original_query = query
        
        if query:
            try:
                # 1. Retrieve evidence
                retrieved_chunks = retrieve_contextual_chunks(query=query)
                
                # 2. Generate final answer
                if not retrieved_chunks:
                    final_answer = "I could not find any relevant information in the document to answer your question."
                else:
                    final_answer = generate_final_answer(query, retrieved_chunks)
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                final_answer = "An error occurred while processing your question. Please try again."

    return render_template("index.html", answer=final_answer, query=original_query)

if __name__ == "__main__":
    
    initialize_rag_system()
    app.run(debug=True, use_reloader=False)
