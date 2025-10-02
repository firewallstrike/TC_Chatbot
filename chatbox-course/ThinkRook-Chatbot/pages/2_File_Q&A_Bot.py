import streamlit as st
import PyPDF2
import io
import google.generativeai as genai





st.sidebar.title("Bot Settings")

with st.sidebar:
    
    model_name = st.selectbox("Select Your Model", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], key="gemini_model")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")


st.title("📝 File Q&A with Crane Brain")
st.caption("Chat with your uploaded article or file.")

uploaded_file = st.file_uploader("Upload your article/file here.", type=("pdf"))

if not gemini_api_key:
    st.info("Please add your Google API key to continue.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name)


# Initializing the session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.article_text = ""


# If a new file is uploaded, read and store the article/file
if uploaded_file:
    article = uploaded_file.read().decode()
    st.session_state.article_text = article

    if not any(msg["role"]=="assistant" for msg in st.session_state.messages):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Article Uploaded! You can now ask me questions about it."
        })

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me something about the article/file."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.article_text:
        chat_prompt = f"""
                        You are an AI assistant helping summarize and explain and article.
                        Here is your article.
                        <article>{st.session_state.article_text}</article>
                        Now my question is: {prompt}
                        """
        
        chat = model.start_chat(history=[])
        response = chat.send_message(chat_prompt)

        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.chat_message("assistant").write(response.text)

    else:
        st.error("Please upload a file or article first.")