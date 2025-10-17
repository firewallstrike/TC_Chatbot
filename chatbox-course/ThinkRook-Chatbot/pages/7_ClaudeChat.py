"""
Crane Brain Claude Bot - A Streamlit chatbot powered by Anthropic's Claude AI

This application creates an interactive chat interface using Streamlit and Anthropic's 
Claude model. Users can have conversations with the AI assistant, 
with full chat history maintained throughout the session.
"""

import streamlit as st
from anthropic import Anthropic

# Set up the page title and caption
st.title("Crane Brain Claude Bot")
st.caption("Hello from Claude!")
st.sidebar.title("AI Bot Settings")

# Initialize the Claude API selection with the provided key
with st.sidebar:
    claude_api_key = st.text_input("Anthropic API Key", key="claude_api_key", type="password")
    st.caption("Get your API key from: [console.anthropic.com](https://console.anthropic.com)")
    
    model = st.selectbox(
        "Select Your Model", 
        [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514", 
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ], 
        key="claude_model"
    )

# Initialize chat history in session state if it doesn't exist
# Session state persists data across reruns of the Streamlit app
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Claude. How can I assist you today?"}]

# Display all previous messages in the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
# The := operator assigns the input value to 'prompt' and checks if it's not empty
if prompt := st.chat_input():

    # If no API key
    if not claude_api_key:
        st.info("Please add your Anthropic API key to continue!")
        st.stop()

    # Initialize the Anthropic client with the provided key
    client = Anthropic(api_key=claude_api_key)
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in the chat interface
    st.chat_message("user").write(prompt)

    # Prepare messages for Claude API (exclude the initial assistant greeting for the API call)
    api_messages = [msg for msg in st.session_state.messages if msg != {"role": "assistant", "content": "Hello! I'm Claude. How can I assist you today?"}]
    
    # Send the conversation to Claude and get response
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=api_messages
        )
        
        # Extract the text from Claude's response
        response_text = response.content[0].text

        # Add assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        # Display assistant's response in the chat interface
        st.chat_message("assistant").write(response_text)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API key and try again.")