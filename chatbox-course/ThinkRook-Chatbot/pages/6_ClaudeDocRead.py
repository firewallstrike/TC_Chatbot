# ============================================================================
# Crane Brain Claude Bot - Improved Version
#
# HOW TO RUN:
# 1. Save this code as a Python file (e.g., `app.py`).
# 2. Install requirements: pip install -r requirements.txt
# 3. Run: streamlit run app.py
# ============================================================================

import streamlit as st
import pandas as pd
import os
import logging
from typing import Optional
from anthropic import Anthropic

# LangChain Imports
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MAX_FILE_SIZE_MB = 10
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_SEARCH_K = 3
    SUPPORTED_DOCS = ["pdf", "docx"]
    SUPPORTED_DATA = ["csv", "xlsx"]
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    TEMP_DIR = "/tmp"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_file_size(uploaded_file, max_size_mb: int = Config.MAX_FILE_SIZE_MB) -> bool:
    """Validate file size before processing."""
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size > max_size_mb:
        raise ValueError(f"File too large: {file_size:.2f}MB (max: {max_size_mb}MB)")
    return True

def cleanup_temp_file(filepath: str):
    """Safely remove temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file: {e}")

# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def get_embeddings(api_key: str):
    """Get cached embeddings model using Voyage AI (recommended for Claude)."""
    try:
        # Try to use Anthropic's embeddings if available
        from langchain_community.embeddings import VoyageEmbeddings
        return VoyageEmbeddings(voyage_api_key=api_key, model="voyage-2")
    except:
        # Fallback to OpenAI embeddings (you'll need an OpenAI key for this)
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

@st.cache_resource
def get_llm(model_name: str, api_key: str, temperature: float = 0.3) -> ChatAnthropic:
    """Get cached LLM instance."""
    return ChatAnthropic(
        model=model_name, 
        temperature=temperature, 
        anthropic_api_key=api_key,
        max_tokens=4096
    )

# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================

def create_rag_processor(uploaded_file, claude_api_key: str) -> Optional[FAISS]:
    """
    Creates a Retrieval-Augmented Generation (RAG) processor for PDF/DOCX files.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        claude_api_key: Valid Anthropic API key
        
    Returns:
        FAISS vector store or None if processing fails
    """
    if uploaded_file is None or not claude_api_key:
        return None
    
    # Validate file size
    try:
        validate_file_size(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return None
        
    # Ensure temp directory exists
    if not os.path.exists(Config.TEMP_DIR):
        os.makedirs(Config.TEMP_DIR)

    temp_filepath = os.path.join(Config.TEMP_DIR, uploaded_file.name)
    
    try:
        # Save uploaded file temporarily
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        logger.info(f"Processing file: {uploaded_file.name}")

        # Load document based on type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(temp_filepath)
        else:
            st.error("Unsupported document type for RAG.")
            return None

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading document...")
        progress_bar.progress(25)
        documents = loader.load()

        # Validate document content
        if not documents:
            st.error("The document loader failed to return any content.")
            st.warning("This can happen if the file is corrupt or completely unreadable.")
            return None

        all_page_content = "".join(doc.page_content for doc in documents)
        if not all_page_content.strip():
            st.error("Failed to extract any text from the document.")
            st.warning(
                "This often happens with scanned image-based PDFs (OCR is not supported), "
                "or files that contain no actual text. Please check the file's content."
            )
            return None

        # Split text into chunks
        status_text.text("Splitting text into chunks...")
        progress_bar.progress(50)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.error("Splitting the document failed unexpectedly.")
            return None

        # Create embeddings and vector store
        status_text.text("Creating embeddings...")
        progress_bar.progress(75)
        
        # For embeddings, we'll use a simpler approach with sentence-transformers
        try:
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            # Fallback for older langchain versions
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            st.info("Install sentence-transformers: pip install sentence-transformers")
            return None
        
        status_text.text("Building vector store...")
        progress_bar.progress(90)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        logger.info(f"Successfully processed: {uploaded_file.name}")
        
        return vector_store

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"An error occurred while processing the document: {e}")
        return None
    
    finally:
        # Always clean up temporary file
        cleanup_temp_file(temp_filepath)


def create_pandas_agent(uploaded_file, llm):
    """
    Creates a Pandas DataFrame Agent to interact with CSV/XLSX data.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        llm: Language model instance
        
    Returns:
        Pandas DataFrame agent or None if creation fails
    """
    if uploaded_file is None:
        return None

    try:
        # Validate file size
        validate_file_size(uploaded_file)
        
        logger.info(f"Creating Pandas agent for: {uploaded_file.name}")
        
        # Load data based on file type
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            return None
        
        # Display warning about code execution
        st.warning("⚠️ The Pandas agent will execute code to analyze your data. Only upload trusted files.")
        
        return create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
    
    except ValueError as e:
        st.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Error creating Pandas agent: {e}")
        st.error(f"Error creating Pandas agent: {e}")
        return None

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'messages': [
            {"role": "assistant", "content": "Hello! I'm powered by Claude. Upload a document (PDF, DOCX) or a data file (CSV, XLSX) and I'll help you analyze it."}
        ],
        'processor': None,
        'document_name': None,
        'file_type': None,
        'processing_complete': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Crane Brain Claude Bot", page_icon="🧠", layout="wide")
    st.title("Crane Brain Claude Bot 📄📊")
    st.caption("Powered by Anthropic Claude | RAG for documents | Pandas Agent for data files")

    # Initialize session state
    init_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input - always require manual entry
        claude_api_key = st.text_input("Anthropic API Key", key="claude_api_key", type="password")
        
        st.caption("Get your API key from: [console.anthropic.com](https://console.anthropic.com)")
        
        model_name = st.selectbox(
            "Select Claude Model", 
            [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022"
            ],
            index=0,  # Default to Claude Sonnet 4
            key="claude_model"
        )
        
        st.divider()
        st.header("📁 File Upload")
        st.caption(f"Max file size: {Config.MAX_FILE_SIZE_MB}MB")
        
        uploaded_file = st.file_uploader(
            "Upload a file", 
            type=Config.SUPPORTED_DOCS + Config.SUPPORTED_DATA
        )
        
        # File processing
        if uploaded_file is not None:
            if st.session_state.document_name != uploaded_file.name:
                if not claude_api_key:
                    st.warning("⚠️ Please enter your Anthropic API key first!")
                else:
                    with st.spinner("Processing file... This may take a moment."):
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.file_type = uploaded_file.type
                        
                        llm = get_llm(model_name, claude_api_key, temperature=0.3)

                        if uploaded_file.type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                            st.session_state.processor = create_rag_processor(uploaded_file, claude_api_key)
                            file_desc = "document"
                        elif uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                            st.session_state.processor = create_pandas_agent(uploaded_file, llm)
                            file_desc = "data file"
                        
                        if st.session_state.processor:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"✅ I've successfully loaded and processed '{uploaded_file.name}' ({file_desc}). Ask me anything about it!"
                            })
                            st.session_state.processing_complete = True
                            st.success(f"✅ File processed: {uploaded_file.name}")
                        else:
                            st.error("❌ Failed to process the file. Please check the warnings above.")
            else:
                st.success(f"✅ Current file: {uploaded_file.name}")

        # Clear file button
        if st.session_state.processor:
            if st.button("🗑️ Clear File", use_container_width=True):
                st.session_state.processor = None
                st.session_state.document_name = None
                st.session_state.file_type = None
                st.session_state.processing_complete = False
                st.session_state.messages.append({"role": "assistant", "content": "File cleared. Upload a new file to continue."})
                st.rerun()

        # Status display
        st.divider()
        st.subheader("📊 Status")
        if st.session_state.processor:
            st.write(f"**File:** {st.session_state.document_name}")
            processor_type = 'RAG (Document)' if st.session_state.file_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'] else 'Pandas Agent (Data)'
            st.write(f"**Processor:** {processor_type}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**Status:** 🟢 Ready for questions")
        else:
            st.write("**Status:** 🔴 No file loaded")

    # Main chat interface
    if not claude_api_key:
        st.info("👈 Please enter your Anthropic API key in the sidebar to begin.")
        st.info("Get your API key from [console.anthropic.com](https://console.anthropic.com)")
        st.stop()

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if st.session_state.processor is None:
            response_text = "Please upload a file first so I can help you analyze it."
            st.chat_message("assistant").write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response_text = ""
                        
                        # RAG processing for documents
                        if st.session_state.file_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                            vector_store = st.session_state.processor
                            llm = get_llm(model_name, claude_api_key, temperature=0.5)
                            
                            docs = vector_store.similarity_search(prompt, k=Config.SIMILARITY_SEARCH_K)
                            chain = load_qa_chain(llm, chain_type="stuff")
                            response = chain.invoke({"input_documents": docs, "question": prompt})
                            response_text = response.get('output_text', 'No answer found.')

                        # Pandas agent processing for data files
                        elif st.session_state.file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                            agent_executor = st.session_state.processor
                            response = agent_executor.invoke({"input": prompt})
                            response_text = response.get('output', 'Could not process the request.')

                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.write(response_text)
                        logger.info(f"Successfully processed query: {prompt[:50]}...")

                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        error_message = f"❌ An error occurred: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.write(error_message)

if __name__ == "__main__":
    main()
