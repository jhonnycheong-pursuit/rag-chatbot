import streamlit as st
import requests
import os
import time

# --- Configuration ---
FLASK_BACKEND_URL = os.environ.get("FLASK_BACKEND_URL", "http://127.0.0.1:5000") # Use env var or default
PROCESS_ENDPOINT = f"{FLASK_BACKEND_URL}/process"
ASK_ENDPOINT = f"{FLASK_BACKEND_URL}/ask"
LIST_FILES_ENDPOINT = f"{FLASK_BACKEND_URL}/list_files"
CHECK_INDEX_ENDPOINT = f"{FLASK_BACKEND_URL}/check_index"

# --- Helper Functions ---
def process_documents(uploaded_files, urls):
    """Sends documents to the backend for processing."""
    files_to_send = []
    if uploaded_files:
        for file in uploaded_files:
            # Prepare file for requests: (filename, file_object, content_type)
            files_to_send.append(('files', (file.name, file.getvalue(), file.type)))

    payload = {'urls': urls}

    try:
        # Use files= for multipart/form-data encoding (needed for file uploads)
        # Use data= for form fields like 'urls'
        response = requests.post(PROCESS_ENDPOINT, files=files_to_send, data=payload, timeout=300) # Increased timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend or processing failed: {e}")
        if e.response is not None:
            st.error(f"Backend response: {e.response.text}")
        return {"status": "error", "message": f"Failed to connect or process: {e}"}
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


def ask_question(question):
    """Sends a question to the backend and gets the answer."""
    payload = {"question": question}
    try:
        response = requests.post(ASK_ENDPOINT, json=payload, timeout=120) # Timeout for potentially long LLM calls
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        if e.response is not None:
             st.error(f"Backend response: {e.response.text}") # Show more detail
        # Provide a more user-friendly error message in the chat
        return {"status": "error", "message": f"Could not reach the backend to answer. Please check if it's running. ({e})"}
    except Exception as e:
        st.error(f"An unexpected error occurred while asking: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


def get_uploaded_files():
    """Gets the list of previously uploaded files from the backend."""
    try:
        response = requests.get(LIST_FILES_ENDPOINT, timeout=30)
        response.raise_for_status()
        return response.json().get("files", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend to get file list: {e}")
        if e.response is not None:
            st.error(f"Backend response: {e.response.text}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while getting file list: {e}")
        return []


def check_index_exists():
    """Checks if the FAISS index exists and is ready for querying."""
    try:
        response = requests.get(CHECK_INDEX_ENDPOINT, timeout=30)
        response.raise_for_status()
        return response.json().get("index_exists", False)
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend to check index: {e}")
        if e.response is not None:
            st.error(f"Backend response: {e.response.text}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while checking index: {e}")
        return False

# --- Streamlit App UI ---

st.set_page_config(page_title="Document RAG Chatbot", layout="wide")
st.title("📄 Document RAG Chatbot")
st.caption("Upload documents (PDF, TXT, JSON) or provide URLs, then ask questions about their content.")

# --- Sidebar for Document Upload and Processing ---
with st.sidebar:
    st.header("1. Add Documents")

    # Tab for new uploads and previously uploaded files
    upload_tab, existing_tab = st.tabs(["Upload New", "Previously Uploaded"])

    # Generate unique keys for widgets if we need to reset them
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    # Check if we need to reset inputs
    if "reset_inputs" in st.session_state and st.session_state.reset_inputs:
        # Clear the reset flag
        st.session_state.reset_inputs = False
        # Increment the counter to generate new widget keys
        st.session_state.reset_counter += 1

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload Files (PDF, TXT, JSON)",
            type=["pdf", "txt", "json"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.reset_counter}"
        )
        st.markdown("---")
        url_input = st.text_area(
            "Or Provide URLs (one per line)",
            height=100,
            key=f"url_input_{st.session_state.reset_counter}"
        )

    with existing_tab:
        # Get previously uploaded files
        existing_files = get_uploaded_files()

        if existing_files:
            st.write(f"Found {len(existing_files)} previously uploaded files:")

            # Create a multiselect for selecting files to process
            file_options = {f"{file['name']} ({file['size_kb']} KB, {file['modified']})": file['path'] for file in existing_files}
            selected_files = st.multiselect(
                "Select files to process:",
                options=list(file_options.keys()),
                key="existing_files_select"
            )

            # Convert selected file names to file paths
            selected_file_paths = [file_options[file_name] for file_name in selected_files]

            if selected_files:
                st.info(f"Selected {len(selected_files)} files to process")
        else:
            st.info("No previously uploaded files found.")

    st.markdown("---")

    # Process button
    if st.button("Process Documents", key="process_button"):
        # Get URLs from input
        # Get the current URL input value directly from the widget
        current_url_input = url_input if url_input else ""
        # Process the URLs
        url_list = [url.strip() for url in current_url_input.replace(',', '\n').splitlines() if url.strip()]
        url_string_for_backend = ",".join(url_list) # Backend expects comma-separated

        # Check if we have any files or URLs to process
        has_new_uploads = uploaded_files and len(uploaded_files) > 0
        has_existing_files = 'existing_files_select' in st.session_state and len(st.session_state.existing_files_select) > 0
        has_urls = len(url_list) > 0

        # No debug information in production

        if not (has_new_uploads or has_existing_files or has_urls):
            st.warning("Please upload at least one file, select an existing file, or provide at least one URL.")
        else:
            with st.spinner("Processing documents... This might take a while depending on size and number."):
                start_time = time.time()

                # Process new uploads and/or URLs
                if has_new_uploads or has_urls:
                    result = process_documents(uploaded_files, url_string_for_backend)
                else:
                    # Set a default result in case we're only processing existing files
                    result = {"status": "success"}

                # If there are selected existing files, we need to process them too
                # Note: The backend already knows about these files, so we don't need to send them again
                if has_existing_files:
                    st.info(f"Processing {len(selected_file_paths)} existing files...")
                    # We can just send the paths as URLs since they're local file paths
                    existing_files_string = ",".join(selected_file_paths)
                    existing_result = process_documents([], existing_files_string)

                    # Merge results
                    if existing_result.get("status") != "success":
                        result = existing_result  # Use the error from processing existing files

                end_time = time.time()
                processing_time = end_time - start_time

                if result.get("status") == "success":
                    st.success(f"Documents processed successfully in {processing_time:.2f} seconds!")
                    st.session_state.documents_processed = True # Flag to enable chat

                    # Set a flag to reset the inputs on next rerun
                    st.session_state.reset_inputs = True

                    # Force a rerun to update the UI
                    st.rerun()
                else:
                    st.error(f"Processing failed: {result.get('message', 'Unknown error')}")
                    st.session_state.documents_processed = False

    if st.session_state.get("documents_processed", False):
         st.success("✅ Documents are ready. Ask questions below.")
    else:
         st.info("Upload/provide documents and click 'Process Documents' to start.")


# --- Main Chat Interface ---
st.header("2. Ask Questions")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if the index exists when the page loads
if "documents_processed" not in st.session_state:
    index_exists = check_index_exists()
    st.session_state.documents_processed = index_exists
    if index_exists:
        st.success("✅ Documents are ready. Ask questions below.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Check if documents have been processed
    if not st.session_state.get("documents_processed", False):
         st.warning("Please process some documents first using the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            # Get answer from backend
            response_data = ask_question(prompt)

            if response_data.get("status") == "success":
                full_response = response_data.get("answer", "Sorry, I couldn't find an answer.")
            else:
                # Use the error message from the backend response
                full_response = response_data.get("message", "An error occurred.")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- How to Run ---
# 1. Save the files as described in the structure.
# 2. Create the .env file with your OpenAI key.
# 3. Install requirements: pip install -r requirements.txt
# 4. Run the Flask backend: python app.py
# 5. In a *separate* terminal, run the Streamlit frontend: streamlit run frontend.py