# ============================================================================
# Crane Brain Gemini Bot - Definitive Working Version
#
# HOW TO RUN:
# 1. Save this code as a Python file (e.g., `app.py`).
# 2. Make sure you have a `requirements.txt` file and have run `pip install -r requirements.txt`.
# 3. Run the Streamlit app from your terminal:
#    streamlit run app.py
# ============================================================================

import streamlit as st
import google.generativeai as genai
import pandas as pd
import os

# LangChain Imports for advanced processing
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# ============================================================================
# ADVANCED FILE PROCESSING FUNCTIONS
# ============================================================================

def create_rag_processor(uploaded_file, gemini_api_key):
    """
    Creates a Retrieval-Augmented Generation (RAG) processor for PDF/DOCX files.
    This version includes more robust checks for empty or unreadable content.
    """
    if uploaded_file is None or not gemini_api_key:
        return None
        
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")

    temp_filepath = os.path.join("/tmp", uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(temp_filepath)
        else:
            st.error("Unsupported document type for RAG.")
            return None

        documents = loader.load()

        # --- NEW ROBUST CHECK ---
        # 1. Check if the loader returned anything at all.
        if not documents:
            st.error("The document loader failed to return any content.")
            st.warning("This can happen if the file is corrupt or completely unreadable.")
            return None

        # 2. Check if the loaded content is just empty strings or whitespace.
        all_page_content = "".join(doc.page_content for doc in documents)
        if not all_page_content.strip():
            st.error("Failed to extract any text from the document.")
            st.warning(
                "This often happens with scanned image-based PDFs (OCR is not supported), "
                "or files that contain no actual text. Please check the file's content."
            )
            return None
        # --- END OF NEW CHECK ---

    except Exception as e:
        st.error(f"An error occurred while loading the document: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        st.error("Splitting the document failed unexpectedly, even though text was found.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=gemini_api_key
        )
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.info("Please ensure your API key is valid and has permissions for the embedding model.")
        return None

    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


def create_pandas_agent(uploaded_file, llm):
    """
    Creates a Pandas DataFrame Agent to interact with CSV/XLSX data.
    """
    if uploaded_file is None:
        return None

    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            return None
            
        return create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
    
    except Exception as e:
        st.error(f"Error creating Pandas agent: {e}")
        return None

# ============================================================================
# STREAMLIT UI SETUP (No changes needed below this line)
# ============================================================================

st.set_page_config(page_title="Crane Brain Gemini Bot", page_icon="🤖", layout="wide")
st.title("Crane Brain Gemini Bot 📄📊 (Advanced)")
st.caption("Now with RAG for documents and a Pandas Agent for data files!")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a document (PDF, DOCX) or a data file (CSV, XLSX) and I'll help you analyze it."}
    ]
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")
    
    model_name = st.selectbox(
        "Select Your Model", 
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        key="gemini_model"
    )
    
    st.divider()
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "csv", "xlsx"])
    
    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            if not gemini_api_key:
                st.warning("Please enter your Gemini API key first!")
            else:
                with st.spinner("Processing file... This may take a moment."):
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.file_type = uploaded_file.type
                    
                    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=gemini_api_key)

                    if uploaded_file.type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                        st.session_state.processor = create_rag_processor(uploaded_file, gemini_api_key)
                        file_desc = "document"
                    elif uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                        st.session_state.processor = create_pandas_agent(uploaded_file, llm)
                        file_desc = "data file"
                    
                    if st.session_state.processor:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"✅ I've successfully loaded and processed '{uploaded_file.name}' ({file_desc}). Ask me anything about it!"
                        })
                        st.success(f"✅ File processed: {uploaded_file.name}")
                    else:
                        st.error("Failed to process the file. Please check the on-screen warnings for more details.")
        else:
            st.success(f"✅ Current file: {uploaded_file.name}")

    if st.session_state.processor:
        if st.button("🗑️ Clear File", use_container_width=True):
            st.session_state.processor = None
            st.session_state.document_name = None
            st.session_state.file_type = None
            st.session_state.messages.append({"role": "assistant", "content": "File cleared."})
            st.rerun()

    st.divider()
    st.subheader("📊 Status")
    if st.session_state.processor:
        st.write(f"**File:** {st.session_state.document_name}")
        st.write(f"**Processor:** {'RAG (Document)' if st.session_state.file_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'] else 'Pandas Agent (Data)'}")
        st.write(f"**Status:** 🟢 Ready for questions")
    else:
        st.write("**Status:** 🔴 No file loaded")

if not gemini_api_key:
    st.info("👈 Please enter your Google Gemini API key in the sidebar to begin.")
    st.stop()

try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Google API: {e}")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.processor is None:
        st.chat_message("assistant").write("Please upload a file first so I can help you with it.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_text = ""
                    if st.session_state.file_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                        vector_store = st.session_state.processor
                        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=gemini_api_key)
                        docs = vector_store.similarity_search(prompt, k=3)
                        chain = load_qa_chain(llm, chain_type="stuff")
                        response = chain.invoke({"input_documents": docs, "question": prompt})
                        response_text = response.get('output_text', 'No answer found.')

                    elif st.session_state.file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                        agent_executor = st.session_state.processor
                        response = agent_executor.invoke({"input": prompt})
                        response_text = response.get('output', 'Could not process the request.')

                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.write(response_text)

                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.write(error_message)
