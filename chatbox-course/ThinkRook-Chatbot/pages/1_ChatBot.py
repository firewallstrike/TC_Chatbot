"""
ThinkRook Gemini Bot - A Streamlit chatbot powered by Google's Gemini AI

This application creates an interactive chat interface using Streamlit and Google's 
Generative AI (Gemini) model. Users can have conversations with the AI assistant, 
with full chat history maintained throughout the session.
"""

import streamlit as st
import google.generativeai as genai

# Set up the page title and caption

st.title("ThinkRook Gemini Bot")
st.caption("Hello from Gemini Bot!")
st.sidebar.title("AI Bot Settings")
# Initialize the Gemini API selection with the provided key
with st.sidebar:
    gemeni_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")
    model = st.selectbox("Select Your Model", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], key="gemini_model")



# Initialize the Gemini API with the provided key
# genai.configure(api_key=api_key)
# Create a GenerativeModel instance with the specified model
# model = genai.GenerativeModel(model_name)

# Initialize chat history in session state if it doesn't exist
# Session state persists data across reruns of the Streamlit app
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display all previous messages in the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
# The := operator assigns the input value to 'prompt' and checks if it's not empty
if prompt := st.chat_input():

    #if no API key
    if not gemeni_api_key:
        st.info("Please add your Google API key to continue!")
        st.stop()

        # Initialize the Gemini API with the provided key
    genai.configure(api_key=gemeni_api_key)
    # Create a GenerativeModel instance with the specified model
    model = genai.GenerativeModel(model)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in the chat interface
    st.chat_message("user").write(prompt)

    # Create a new chat session with empty history
    chat = model.start_chat(history=[])
    # Send the user's message to the Gemini model and get response
    response = chat.send_message(prompt)

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    # Display assistant's response in the chat interface
    st.chat_message("assistant").write(response.text)
