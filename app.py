import faiss
import pickle
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, session
import os
from dotenv import load_dotenv

import config

from data_loader import load_from_source, add_drive_chunks_to_index, yahoo_llm_answer, jira_llm_answer
from create_embed_doc import extract_text, embed_chunks, create_faiss_index


from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "secret123")

faiss_index = None
text_chunks = None

# ------------------------
# INIT RAG SYSTEM
# ------------------------
def initialize_rag_system():
    global faiss_index, text_chunks
    if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE):
        faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
        with open(config.METADATA_FILE, 'rb') as f:
            text_chunks = pickle.load(f)
    else:
        all_chunks = []
        for source in config.DATA_SOURCES:
            chunks = load_from_source(source)
            if chunks:
                all_chunks.extend(chunks)
        embeddings = embed_chunks(all_chunks, title=config.DOCUMENT_TITLE)
        faiss_index = create_faiss_index(embeddings, all_chunks)
        text_chunks = all_chunks

# ------------------------
# RETRIEVAL FUNCTIONS
# ------------------------
def retrieve_contextual_chunks(query, top_k=11, window_size=4):
    global faiss_index, text_chunks
    if faiss_index is None or text_chunks is None:
        raise ValueError("FAISS index not loaded")

    result = genai.embed_content(
        model="models/embedding-001", 
        content=query, 
        task_type="retrieval_query"
    )
    query_embedding = np.array([result['embedding']]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)

    center_index = indices[0][0]
    start_index = max(0, center_index - window_size)
    end_index = min(len(text_chunks) - 1, center_index + window_size)

    contextual_indices = list(range(start_index, end_index + 1))
    retrieved_chunks = [text_chunks[i] for i in contextual_indices]
    return retrieved_chunks

def generate_final_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt_template = f"""
    DOCUMENT CONTEXT:
    {context}

    USER QUESTION: {query}

    INSTRUCTIONS:
    1. Use ONLY the above context for answering
    2. Return HTML formatted answer
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt_template)
    return response.text

# ------------------------
# ROUTES
# ------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    final_answer = ""
    original_query = ""

    if request.method == "POST":
        query = request.form.get("query")
        original_query = query

        try:
            # Detect Yahoo query
            if "stock" in query.lower() or "price" in query.lower() or "market" in query.lower():
                final_answer = yahoo_llm_answer(query, generate_final_answer)
            
            # Detect Jira query
            elif "jira" in query.lower() or "project" in query.lower() or "sprint" in query.lower():
                final_answer = jira_llm_answer(query, generate_final_answer)
            
            # Default: FAISS RAG
            else:
                retrieved_chunks = retrieve_contextual_chunks(query=query)
                if not retrieved_chunks:
                    final_answer = "No relevant info found."
                else:
                    final_answer = generate_final_answer(query, retrieved_chunks)
                    

        except Exception as e:
            print(f"Error processing query: {e}")
            final_answer = "Error occurred while processing."

    return render_template("index.html", answer=final_answer, query=original_query)

# ------------------------
# GOOGLE DRIVE OAUTH FLOW
# ------------------------
@app.route("/auth/drive")
def auth_drive():
    flow = Flow.from_client_secrets_file(
        "credentials.json",
        scopes=['https://www.googleapis.com/auth/drive.readonly'],
        redirect_uri=url_for('drive_callback', _external=True)
    )
    auth_url, state = flow.authorization_url(prompt='consent')
    session['state'] = state  # store only state
    return redirect(auth_url)

@app.route("/drive/callback")
def drive_callback():
    flow = Flow.from_client_secrets_file(
        "credentials.json",
        scopes=['https://www.googleapis.com/auth/drive.readonly'],
        redirect_uri=url_for('drive_callback', _external=True)
    )
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    with open("token.json", "w") as token:
        token.write(creds.to_json())
    return redirect(url_for("list_drive_files"))

@app.route("/drive/files")
def list_drive_files():
    creds = Credentials.from_authorized_user_file("token.json")
    service = build('drive', 'v3', credentials=creds)

    results = service.files().list(
        pageSize=100,  # increase limit
        fields="files(id, name, mimeType)",
        orderBy="modifiedTime desc",  # newest first
        q="mimeType='application/pdf' or mimeType='application/vnd.google-apps.document'" 
    ).execute()

    files = results.get('files', [])
    return render_template("drive_files.html", files=files)

@app.route("/add_drive_file/<file_id>")
def add_drive_file(file_id):
    add_drive_chunks_to_index(file_id, config.FAISS_INDEX_FILE, config.METADATA_FILE)

    global faiss_index, text_chunks
    faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
    with open(config.METADATA_FILE, 'rb') as f:
        text_chunks = pickle.load(f)

    return redirect(url_for("home"))


if __name__ == "__main__":
    initialize_rag_system()
    app.run(debug=True, use_reloader=False)
