import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    WebBaseLoader,
)
from langchain.docstore.document import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (especially OpenAI API key)
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"
# Make sure FAISS_INDEX_PATH directory exists
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# --- Document Loading ---
def load_document(source):
    """Loads a document from a file path or URL."""
    logger.info(f"Attempting to load source: {source}")
    try:
        # Check if it's a URL (with or without http/https prefix)
        if source.startswith("http://") or source.startswith("https://") or source.startswith("www."):
            # Add http:// prefix if missing
            if not (source.startswith("http://") or source.startswith("https://")):
                source = "https://" + source
                logger.info(f"Added https:// prefix to URL: {source}")
            loader = WebBaseLoader(source)
            return loader.load()
        elif source.endswith(".pdf"):
            loader = PyPDFLoader(source)
            return loader.load_and_split() # PyPDFLoader can split pages
        elif source.endswith(".txt"):
            loader = TextLoader(source, encoding='utf-8') # Specify encoding
            return loader.load()
        elif source.endswith(".json"):
            try:
                # First, try to load the JSON file directly
                with open(source, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Convert the JSON to a string representation for processing
                json_str = json.dumps(json_data, indent=2)

                # Create a single document with the formatted JSON content
                doc = Document(page_content=json_str, metadata={"source": source})
                logger.info(f"Successfully loaded JSON file: {source}")
                return [doc]
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file {source}: {e}")
                return []
            except Exception as e:
                # Fallback to JSONLoader if direct loading fails
                logger.warning(f"Falling back to JSONLoader for {source}: {e}")
                try:
                    # Try with a more flexible jq_schema
                    loader = JSONLoader(source, jq_schema='.', text_content=False)
                    docs = loader.load()
                    if docs:
                        logger.info(f"Successfully loaded JSON file with JSONLoader: {source}")
                        return docs
                    else:
                        logger.warning(f"JSONLoader returned no documents for {source}")
                        return []
                except Exception as e2:
                    logger.error(f"JSONLoader also failed for {source}: {e2}")
                    return []
        else:
            logger.warning(f"Unsupported source type: {source}. Skipping.")
            return []
    except Exception as e:
        logger.error(f"Error loading source {source}: {e}")
        return []

def load_multiple_sources(sources):
    """Loads documents from a list of file paths or URLs."""
    all_docs = []
    for source in sources:
        loaded_docs = load_document(source)
        if loaded_docs:
            all_docs.extend(loaded_docs)
            logger.info(f"Successfully loaded {len(loaded_docs)} documents from {source}")
        else:
            logger.warning(f"Could not load any documents from {source}")
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs

# --- Text Splitting ---
def split_documents(documents):
    """Splits loaded documents into smaller chunks."""
    if not documents:
        logger.warning("No documents provided to split.")
        return []
    # Use a larger chunk size and overlap for more context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks to capture more context
        chunk_overlap=300,  # More overlap to avoid losing context between chunks
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Try to split on paragraph, then sentence
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# --- Embedding and Vector Store ---
def get_embeddings_model():
    """Initializes the OpenAI embeddings model."""
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise

def create_and_save_faiss_index(chunks):
    """Creates a FAISS index from document chunks and saves it."""
    if not chunks:
        logger.warning("No chunks provided to create index.")
        return False
    try:
        embeddings = get_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS index created and saved to {FAISS_INDEX_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to create or save FAISS index: {e}")
        return False

def check_index_exists():
    """Checks if a valid FAISS index exists."""
    # Check if the index directory exists
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(f"FAISS index directory not found at {FAISS_INDEX_PATH}")
        return False

    # Check if the index files exist
    index_files = os.listdir(FAISS_INDEX_PATH)
    if not index_files:
        logger.warning(f"FAISS index directory is empty at {FAISS_INDEX_PATH}")
        return False

    # Check for required index files
    required_files = ['index.faiss', 'index.pkl']
    for file in required_files:
        if file not in index_files:
            logger.warning(f"Required index file {file} not found in {FAISS_INDEX_PATH}")
            return False

    logger.info(f"Found valid FAISS index with files: {', '.join(index_files)}")
    return True


def load_faiss_index():
    """Loads an existing FAISS index."""
    # Check if the index directory exists
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(f"FAISS index directory not found at {FAISS_INDEX_PATH}")
        return None

    # Check if the index files exist
    index_files = os.listdir(FAISS_INDEX_PATH)
    if not index_files:
        logger.warning(f"FAISS index directory is empty at {FAISS_INDEX_PATH}")
        return None
    else:
        logger.info(f"Found FAISS index files: {', '.join(index_files)}")

    try:
        embeddings = get_embeddings_model()
        # FAISS requires explicit permission for deserialization
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Add this line
        )

        # Log more information about the loaded index
        if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
            logger.info(f"FAISS index loaded successfully with {vector_store.index.ntotal} vectors")
        else:
            logger.info(f"FAISS index loaded successfully from {FAISS_INDEX_PATH}")

        return vector_store
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        # Consider deleting the potentially corrupt index directory here if loading fails consistently
        # import shutil
        # shutil.rmtree(FAISS_INDEX_PATH)
        return None

# --- QA Chain ---
def get_qa_chain(vector_store):
    """Creates the RetrievalQA chain."""
    if vector_store is None:
        logger.error("Cannot create QA chain without a valid vector store.")
        return None
    try:
        from langchain.prompts import PromptTemplate

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # Or "gpt-4"

        # Configure the retriever to use MMR for more diverse results
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 8,  # Return 8 documents
                "fetch_k": 20,  # Fetch 20 documents and then pick the most diverse 8
                "lambda_mult": 0.7  # 0.7 diversity vs 0.3 relevance balance
            }
        )
        logger.info("Configured retriever to use MMR with 8 diverse document chunks")

        # Create a custom prompt that instructs the model to prioritize the document content
        custom_prompt = PromptTemplate(
            template="""You are a helpful assistant answering questions about documents that have been uploaded.

Your primary goal is to answer questions based on the information in the provided context.
The context contains text from documents that have been uploaded and processed.

Guidelines:
1. If the answer is clearly in the context, provide it directly, citing specific details from the context.
2. If the context contains partial information, use that information as much as possible.
3. If the answer cannot be found in the context at all, say 'I don't have information about that in the documents you've uploaded.'
4. Be helpful and informative while staying grounded in the provided context.
5. If the context contains relevant information but doesn't directly answer the question, explain what information is available.

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Other options: "map_reduce", "refine", "map_rerank"
            retriever=retriever,
            return_source_documents=True, # Optional: to see which chunks were used
            chain_type_kwargs={
                "prompt": custom_prompt
            }
        )
        logger.info("RetrievalQA chain created successfully.")
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        return None

# --- Combined Processing Function ---
def process_and_index_sources(sources):
    """Loads, splits, and indexes documents from sources."""
    logger.info("Starting document processing and indexing...")
    documents = load_multiple_sources(sources)
    if not documents:
        logger.error("No documents were successfully loaded. Aborting indexing.")
        return False
    chunks = split_documents(documents)
    if not chunks:
        logger.error("Document splitting resulted in no chunks. Aborting indexing.")
        return False
    success = create_and_save_faiss_index(chunks)
    if success:
        logger.info("Document processing and indexing completed successfully.")
    else:
        logger.error("Document processing and indexing failed.")
    return success

# --- Query Function ---
def answer_query(question):
    """Answers a query using the loaded FAISS index and QA chain."""
    logger.info(f"Received query: {question}")
    vector_store = load_faiss_index()
    if vector_store is None:
        return "Error: Vector index not found or could not be loaded. Please process documents first."

    qa_chain = get_qa_chain(vector_store)
    if qa_chain is None:
        return "Error: Could not create the Question Answering chain."

    try:
        result = qa_chain({"query": question})
        answer = result.get("result", "Could not generate an answer.")

        # Include source documents for logging only (not in the answer)
        sources = result.get("source_documents", [])

        # Log the retrieved documents for debugging
        logger.info(f"Retrieved {len(sources)} documents for query '{question}'")

        # Log more detailed information about each retrieved document
        if sources:
            for i, doc in enumerate(sources):
                # Log document metadata
                source_info = doc.metadata.get('source', 'Unknown')
                logger.info(f"Document {i+1} source: {source_info}")

                # Log a longer snippet of the content
                content_preview = doc.page_content[:300].replace('\n', ' ')
                logger.info(f"Document {i+1} content: {content_preview}...")

                # Calculate and log similarity score if available
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    logger.info(f"Document {i+1} similarity score: {doc.metadata['score']:.4f}")
        else:
            logger.warning(f"No source documents retrieved for query '{question}'")
            logger.warning("This may indicate that the query is not related to the uploaded documents or the retrieval system needs adjustment.")

        logger.info(f"Generated answer for query '{question}'")
        return answer
    except Exception as e:
        logger.error(f"Error during query execution: {e}")
        return f"An error occurred while processing your question: {e}"