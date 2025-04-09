import os
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
import rag_core # Import our RAG logic module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoints ---

@app.route('/process', methods=['POST'])
def process_documents_endpoint():
    """
    Endpoint to receive documents (files/URLs) and trigger processing/indexing.
    Expects 'files' for uploaded files and 'urls' (comma-separated) in form data.
    """
    logger.info("Received request to /process endpoint")
    sources = []
    uploaded_files_paths = []

    # 1. Handle File Uploads
    if 'files' in request.files:
        files = request.files.getlist('files')
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    sources.append(filepath)
                    uploaded_files_paths.append(filepath)
                    logger.info(f"Saved uploaded file: {filepath}")
                except Exception as e:
                    logger.error(f"Error saving file {filename}: {e}")
            elif file and file.filename:
                 logger.warning(f"File type not allowed: {file.filename}")


    # 2. Handle URLs
    urls_string = request.form.get('urls', '')
    if urls_string:
        urls = [url.strip() for url in urls_string.split(',') if url.strip()]
        sources.extend(urls)
        logger.info(f"Received URLs: {urls}")

    if not sources:
        logger.warning("No valid files or URLs provided.")
        return jsonify({"status": "error", "message": "No valid files or URLs provided."}), 400

    # 3. Trigger Processing in rag_core
    logger.info(f"Processing sources: {sources}")
    success = rag_core.process_and_index_sources(sources)

    # 4. Keep uploaded files instead of removing them
    if uploaded_files_paths:
        logger.info(f"Keeping uploaded files: {', '.join(uploaded_files_paths)}")

    # 5. Respond
    if success:
        logger.info("Processing successful.")
        return jsonify({"status": "success", "message": "Documents processed and indexed successfully."})
    else:
        logger.error("Processing failed.")
        return jsonify({"status": "error", "message": "Failed to process documents."}), 500


@app.route('/ask', methods=['POST'])
def ask_question_endpoint():
    """
    Endpoint to receive a question and return an answer based on indexed documents.
    Expects JSON payload: {"question": "Your question here"}
    """
    logger.info("Received request to /ask endpoint")
    if not request.is_json:
        logger.warning("Request is not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question')

    if not question:
        logger.warning("No question provided in the request.")
        return jsonify({"status": "error", "message": "No question provided."}), 400

    logger.info(f"Asking question: {question}")
    answer = rag_core.answer_query(question)

    # Check if the answer indicates an error from rag_core
    if answer.startswith("Error:"):
         logger.error(f"Error generating answer: {answer}")
         status_code = 500 if "Could not create" in answer else 400 # 400 if index not ready, 500 otherwise
         return jsonify({"status": "error", "message": answer}), status_code
    else:
         logger.info("Answer generated successfully.")
         return jsonify({"status": "success", "answer": answer})


@app.route('/list_files', methods=['GET'])
def list_files_endpoint():
    """
    Endpoint to list all files in the uploads directory.
    """
    logger.info("Received request to /list_files endpoint")

    # Get all files in the uploads directory
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                # Get file size and last modified time
                file_stats = os.stat(filepath)
                size_kb = file_stats.st_size / 1024  # Convert to KB
                modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                files.append({
                    "name": filename,
                    "path": filepath,
                    "size_kb": round(size_kb, 2),
                    "modified": modified_time
                })

        logger.info(f"Found {len(files)} files in uploads directory")
        return jsonify({"status": "success", "files": files})
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"status": "error", "message": f"Failed to list files: {e}"}), 500


@app.route('/check_index', methods=['GET'])
def check_index_endpoint():
    """
    Endpoint to check if the FAISS index exists and is ready for querying.
    """
    logger.info("Received request to /check_index endpoint")

    # Check if the FAISS index exists and is valid
    index_exists = rag_core.check_index_exists()

    if index_exists:
        logger.info("FAISS index exists and is ready for querying")
        return jsonify({"status": "success", "index_exists": True})
    else:
        logger.info("FAISS index does not exist or is not valid")
        return jsonify({"status": "success", "index_exists": False})


if __name__ == '__main__':
    # Make sure to run with host='0.0.0.0' to be accessible on your network
    # if running Streamlit on a different machine or container.
    # Use debug=True only for development.
    app.run(host='0.0.0.0', port=5000, debug=False)