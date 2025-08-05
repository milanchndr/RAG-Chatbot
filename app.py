from flask import Flask, render_template, request
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from create_embed_doc import extract_text, embed_chunks, create_faiss_index


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and metadata
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "finance.pdf")
faiss_index_file = os.path.join(current_dir, "faiss_index.index")
metadata_file = os.path.join(current_dir, "metadata.pkl")

# Create FAISS index and metadata if they don't exist
if not os.path.exists(faiss_index_file) or not os.path.exists(metadata_file):
    print("FAISS index or metadata not found. Creating new ones...")
    
    # Extract text from PDF
    text_chunks = extract_text(pdf_path)
    print(f"Extracted {len(text_chunks)} text chunks")
    
    # Embed text chunks
    embeddings = embed_chunks(text_chunks, title="finance")  # Added title parameter
    print(f"Created {len(embeddings)} embeddings")
    
    # Create FAISS index (this function already saves the index and metadata)
    faiss_index = create_faiss_index(embeddings,text_chunks)
    
    print("FAISS index and metadata created successfully!")
    
else:
    print("FAISS index and metadata already exist. Loading them...")
    
    try:
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_index_file)
        print(f"FAISS index loaded with {faiss_index.ntotal} vectors")
        
        # Load metadata (text chunks)
        with open(metadata_file, 'rb') as f:
            text_chunks = pickle.load(f)
        print(f"Metadata loaded with {len(text_chunks)} chunks")
        
    except Exception as e:
        print(f"Error loading existing files: {e}")
        print("Creating new index and metadata...")
        
        # If loading fails, create new ones
        text_chunks = extract_text(pdf_path)
        embeddings = embed_chunks(text_chunks, title="finance")
        faiss_index = create_faiss_index(embeddings, text_chunks)        
        
        
        
# Global variables to store loaded data
faiss_index = None
text_chunks = None

def load_faiss_and_metadata():
    """Load FAISS index and metadata at startup"""
    global faiss_index, text_chunks
    
    try:
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_index_file)
        print(f"FAISS index loaded successfully with {faiss_index.ntotal} vectors")
        
        # Load metadata (text chunks)
        with open(metadata_file, 'rb') as f:
            text_chunks = pickle.load(f)
        print(f"Metadata loaded successfully with {len(text_chunks)} chunks")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading files: {e}")
        raise

def retrieve_contextual_chunks(query, top_k=5, window_size=5):
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
    
    # Generate the content
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
    load_faiss_and_metadata()
    app.run(debug=True)