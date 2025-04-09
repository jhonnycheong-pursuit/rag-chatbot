# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents or provided URLs.

## Features

- Upload PDF, TXT, and JSON files
- Provide URLs for web content
- Process and index documents using FAISS vector database
- Ask questions about the content of the documents
- Responsive UI with Streamlit

## Architecture

The application consists of two main components:

1. **Backend (Flask)**: Handles document processing, indexing, and question answering
2. **Frontend (Streamlit)**: Provides a user-friendly interface for uploading documents and asking questions

## How It Works

1. **Document Processing**: Documents are loaded, split into chunks, and embedded using OpenAI embeddings
2. **Vector Database**: The embeddings are stored in a FAISS index for efficient similarity search
3. **Question Answering**: When a question is asked, the system:
   - Embeds the question
   - Retrieves relevant document chunks
   - Sends the question and context to an LLM (GPT-3.5-turbo)
   - Returns the generated answer

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

1. Start the Flask backend:
   ```
   python app.py
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```
   streamlit run frontend.py
   ```

3. Open your browser and navigate to http://localhost:8501

## Usage

1. Upload documents or provide URLs in the sidebar
2. Click "Process Documents" to index the content
3. Ask questions in the chat interface
4. View the answers generated based on the document content

## License

[MIT License](LICENSE)
