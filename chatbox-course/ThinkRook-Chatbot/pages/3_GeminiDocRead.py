"""
ThinkRook Gemini Bot with File Upload - A Streamlit chatbot with document analysis

This application creates an interactive chat interface using Streamlit and Google's 
Generative AI (Gemini) model. Users can upload PDF, DOCX, CSV, or XLSX files and 
ask questions about the content, with full chat history maintained throughout the session.
"""

import streamlit as st
import google.generativeai as genai
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
# STREAMLIT UI SETUP
# ============================================================================

# Set up the page configuration
st.set_page_config(page_title="ThinkRook Gemini Bot", page_icon="🤖", layout="wide")

st.title("ThinkRook Gemini Bot 📄📊")
st.caption("Chat with Gemini AI and analyze your documents and data files!")

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
    
    # API Key input
    gemini_api_key = st.text_input(
        "Gemini API Key", 
        key="gemini_api_key", 
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    # Model selection dropdown
    model_name = st.selectbox(
        "Select Your Model", 
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        key="gemini_model",
        help="Choose which Gemini model to use"
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
    st.caption("⚠️ Remember: API keys should be stored securely in production.")

# ============================================================================
# API CONFIGURATION - Check for API key before proceeding
# ============================================================================

# Stop execution if no API key is provided
if not gemini_api_key:
    st.info("👈 Please enter your Google Gemini API key in the sidebar to continue.")
    st.markdown("""
    ### How to get your API key:
    1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    2. Sign in with your Google account
    3. Click "Create API Key"
    4. Copy and paste it in the sidebar
    """)
    st.stop()

# Configure the Gemini API with the provided key
genai.configure(api_key=gemini_api_key)
# Create a GenerativeModel instance with the selected model
model = genai.GenerativeModel(model_name)

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
    
    # Prepare the context for Gemini
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
    
    # Generate response from Gemini
    with st.spinner("Thinking..."):
        try:
            # Create a new chat session
            chat = model.start_chat(history=[])
            # Send the prompt to the Gemini model
            response = chat.send_message(context_prompt)
            
            # Add assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            # Display assistant's response in the chat interface
            st.chat_message("assistant").write(response.text)
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)