# PDF RAG (Retrieval-Augmented Generation) Application

A Flask-based web application that uses FAISS for vector similarity search and Google's Gemini AI to answer questions based on PDF document content.

## Features

- 📄 **PDF Text Extraction**: Automatically extracts and chunks text from PDF documents
- 🔍 **Vector Search**: Uses FAISS for efficient similarity search
- 🤖 **AI-Powered Answers**: Leverages Google Gemini AI for contextual responses
- 🌐 **Web Interface**: Simple and clean web interface for querying documents
- ⚡ **Auto-Indexing**: Automatically creates FAISS index if it doesn't exist
- 📁 **Flexible File Paths**: Works from any directory with proper file structure

## Project Structure

```
├── app.py                 # Main Flask application
├── create_embed_doc.py    # PDF processing and embedding functions
├── templates/
│   └── index.html        # Web interface template
├── finance.pdf           # Your PDF document
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md            # This file