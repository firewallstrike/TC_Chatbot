"""
ThinkRook Ollama Bot with File Upload - A Streamlit chatbot with document analysis

This application creates an interactive chat interface using Streamlit and Ollama 
(local LLM). Users can upload PDF, DOCX, CSV, or XLSX files and ask questions
about the content, with full chat history maintained throughout the session.
"""

import streamlit as st
import requests
import json
import PyPDF2
from docx import Document
import pandas as pd
import io

# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================

def read_pdf(file):
    """
    Extract text content from a PDF file.
    
    Args:
        file: A file-like object containing PDF data
        
    Returns:
        str: Extracted text from all pages of the PDF
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    
    # Iterate through all pages and extract text
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text


def read_docx(file):
    """
    Extract text content from a DOCX file.
    
    Args:
        file: A file-like object containing DOCX data
        
    Returns:
        str: Extracted text from all paragraphs in the document
    """
    doc = Document(file)
    text = ""
    
    # Extract text from each paragraph
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    return text


def read_csv(file):
    """
    Extract data from a CSV file and convert to formatted text.
    
    Args:
        file: A file-like object containing CSV data
        
    Returns:
        str: Formatted text representation of the CSV data with summary statistics
    """
    try:
        # Read CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        
        # Create a comprehensive text representation
        text = "CSV FILE ANALYSIS\n"
        text += "=" * 50 + "\n\n"
        
        # Basic information
        text += f"Number of Rows: {len(df)}\n"
        text += f"Number of Columns: {len(df.columns)}\n\n"
        
        # Column names and data types
        text += "COLUMNS AND DATA TYPES:\n"
        text += "-" * 50 + "\n"
        for col in df.columns:
            text += f"- {col}: {df[col].dtype}\n"
        text += "\n"
        
        # First few rows
        text += "FIRST 10 ROWS:\n"
        text += "-" * 50 + "\n"
        text += df.head(10).to_string() + "\n\n"
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += "SUMMARY STATISTICS (Numeric Columns):\n"
            text += "-" * 50 + "\n"
            text += df[numeric_cols].describe().to_string() + "\n\n"
        
        # Value counts for categorical columns (first 3 columns with <20 unique values)
        categorical_cols = df.select_dtypes(include=['object']).columns
        text += "SAMPLE VALUES (Categorical Columns):\n"
        text += "-" * 50 + "\n"
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            unique_count = df[col].nunique()
            if unique_count < 20:  # Only show if fewer than 20 unique values
                text += f"\n{col} (Value Counts):\n"
                text += df[col].value_counts().head(10).to_string() + "\n"
        
        return text
    
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"


def read_xlsx(file):
    """
    Extract data from an XLSX (Excel) file and convert to formatted text.
    
    Args:
        file: A file-like object containing XLSX data
        
    Returns:
        str: Formatted text representation of the XLSX data with summary statistics
    """
    try:
        # Read Excel file - get all sheet names first
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        
        text = "EXCEL FILE ANALYSIS\n"
        text += "=" * 50 + "\n\n"
        text += f"Number of Sheets: {len(sheet_names)}\n"
        text += f"Sheet Names: {', '.join(sheet_names)}\n\n"
        
        # Process each sheet (limit to first 3 sheets for performance)
        for sheet_name in sheet_names[:3]:
            df = pd.read_excel(file, sheet_name=sheet_name)
            
            text += f"\n{'=' * 50}\n"
            text += f"SHEET: {sheet_name}\n"
            text += f"{'=' * 50}\n\n"
            
            # Basic information
            text += f"Number of Rows: {len(df)}\n"
            text += f"Number of Columns: {len(df.columns)}\n\n"
            
            # Column names and data types
            text += "COLUMNS AND DATA TYPES:\n"
            text += "-" * 50 + "\n"
            for col in df.columns:
                text += f"- {col}: {df[col].dtype}\n"
            text += "\n"
            
            # First few rows
            text += "FIRST 10 ROWS:\n"
            text += "-" * 50 + "\n"
            text += df.head(10).to_string() + "\n\n"
            
            # Summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text += "SUMMARY STATISTICS (Numeric Columns):\n"
                text += "-" * 50 + "\n"
                text += df[numeric_cols].describe().to_string() + "\n\n"
        
        if len(sheet_names) > 3:
            text += f"\n(Note: Only showing first 3 sheets. Total sheets: {len(sheet_names)})\n"
        
        return text
    
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"


def read_uploaded_file(uploaded_file):
    """
    Read and extract text from an uploaded file (PDF, DOCX, CSV, or XLSX).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text content, or None if file type is unsupported
    """
    if uploaded_file is not None:
        # Check file type and process accordingly
        if uploaded_file.type == "application/pdf":
            return read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx(uploaded_file)
        elif uploaded_file.type == "text/csv":
            return read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return read_xlsx(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, CSV, or XLSX file.")
            return None
    return None

# ============================================================================
# OLLAMA API FUNCTIONS
# ============================================================================

def get_available_models(ollama_url):
    """
    Fetch the list of available models from Ollama.
    
    Args:
        ollama_url: Base URL for the Ollama API
        
    Returns:
        list: List of available model names, or empty list if error
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []


def send_message_to_ollama(ollama_url, model_name, prompt):
    """
    Send a message to Ollama and get a response.
    
    Args:
        ollama_url: Base URL for the Ollama API
        model_name: Name of the model to use
        prompt: The prompt to send to the model
        
    Returns:
        str: The model's response, or error message
    """
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120  # 2 minute timeout for large responses
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be processing a large amount of data."
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# STREAMLIT UI SETUP
# ============================================================================

# Set up the page configuration
st.set_page_config(page_title="ThinkRook Ollama Bot", page_icon="🤖", layout="wide")

st.title("ThinkRook Ollama Bot 📄📊")
st.caption("Chat with Ollama (Local LLM) and analyze your documents and data files!")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today? You can upload a document (PDF, DOCX) or data file (CSV, XLSX) and ask me questions about it!"}
    ]

# Initialize document content storage
if 'document_content' not in st.session_state:
    st.session_state.document_content = None

# Initialize document name storage
if 'document_name' not in st.session_state:
    st.session_state.document_name = None

# Initialize file type storage
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# ============================================================================
# SIDEBAR - API CONFIGURATION & FILE UPLOAD
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama URL input
    ollama_url = st.text_input(
        "Ollama API URL", 
        value="http://localhost:11434",
        key="ollama_url",
        help="Enter your Ollama server URL (default: http://localhost:11434)"
    )
    
    # Test connection and fetch models
    if st.button("🔄 Refresh Models", use_container_width=True):
        with st.spinner("Fetching available models..."):
            available_models = get_available_models(ollama_url)
            if available_models:
                st.session_state.available_models = available_models
                st.success(f"Found {len(available_models)} models!")
            else:
                st.error("Could not connect to Ollama. Make sure Ollama is running.")
    
    # Get available models
    if 'available_models' not in st.session_state:
        available_models = get_available_models(ollama_url)
        st.session_state.available_models = available_models if available_models else ["llama3.2", "mistral", "phi3"]
    
    # Model selection dropdown
    model_name = st.selectbox(
        "Select Your Model", 
        st.session_state.available_models,
        key="ollama_model",
        help="Choose which Ollama model to use"
    )
    
    st.divider()
    st.header("📁 File Upload")
    
    # File uploader widget - now supports CSV and XLSX
    uploaded_file = st.file_uploader(
        "Upload a file", 
        type=["pdf", "docx", "csv", "xlsx"],
        help="Upload a PDF, DOCX, CSV, or XLSX file to analyze"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("Processing file..."):
                # Extract text from the file
                text_content = read_uploaded_file(uploaded_file)
                
                if text_content:
                    # Store document content in session state
                    st.session_state.document_content = text_content
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.file_type = uploaded_file.type
                    
                    # Determine file type for message
                    if uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                        file_desc = "data file"
                    else:
                        file_desc = "document"
                    
                    # Add system message about document upload
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"✅ I've successfully loaded '{uploaded_file.name}' ({file_desc}). You can now ask me questions about this file!"
                    })
                    
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    st.info(f"📊 Content length: {len(text_content)} characters")
        else:
            st.success(f"✅ Current file: {uploaded_file.name}")
            st.info(f"📊 Content length: {len(st.session_state.document_content)} characters")
    
    # Show option to clear document
    if st.session_state.document_content:
        if st.button("🗑️ Clear File", use_container_width=True):
            st.session_state.document_content = None
            st.session_state.document_name = None
            st.session_state.file_type = None
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "File cleared. You can upload a new file or continue chatting!"
            })
            st.rerun()
    
    # Display current status
    st.divider()
    st.subheader("📊 Status")
    if st.session_state.document_content:
        st.write(f"**File:** {st.session_state.document_name}")
        
        # Show file type icon
        if st.session_state.file_type == "text/csv":
            st.write(f"**Type:** 📊 CSV Data")
        elif st.session_state.file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            st.write(f"**Type:** 📈 Excel Data")
        elif st.session_state.file_type == "application/pdf":
            st.write(f"**Type:** 📄 PDF Document")
        else:
            st.write(f"**Type:** 📝 Word Document")
            
        st.write(f"**Status:** 🟢 Ready for questions")
    else:
        st.write("**Status:** 🔴 No file loaded")
    
    # Footer
    st.divider()
    st.caption("💡 Tip: Upload data files (CSV/XLSX) for analysis or documents (PDF/DOCX) for content questions!")
    st.caption("⚠️ Make sure Ollama is running locally on your machine.")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Display all previous messages in the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in the chat interface
    st.chat_message("user").write(prompt)
    
    # Prepare the context for Ollama
    # If a document is loaded, include it in the prompt
    if st.session_state.document_content:
        # Customize prompt based on file type
        if st.session_state.file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            context_prompt = f"""You are an AI assistant helping to analyze a data file (CSV or Excel). 

DATA FILE CONTENT:
---
{st.session_state.document_content}
---

USER QUESTION: {prompt}

Please answer the user's question based on the data above. You can:
- Provide insights about the data structure
- Calculate statistics or summaries
- Identify trends or patterns
- Answer specific questions about values or relationships
- Make recommendations based on the data

If the question cannot be answered using the data, let the user know and provide general information if applicable."""
        else:
            context_prompt = f"""You are an AI assistant helping to analyze a document. 

DOCUMENT CONTENT:
---
{st.session_state.document_content}
---

USER QUESTION: {prompt}

Please answer the user's question based on the document content above. If the question cannot be answered using the document, let the user know and provide general information if applicable."""
    else:
        context_prompt = prompt
    
    # Generate response from Ollama
    with st.spinner("Thinking..."):
        try:
            # Send the prompt to the Ollama model
            response_text = send_message_to_ollama(ollama_url, model_name, context_prompt)
            
            # Add assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            # Display assistant's response in the chat interface
            st.chat_message("assistant").write(response_text)
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)
    
