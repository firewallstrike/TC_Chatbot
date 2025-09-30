import streamlit as st 

st.title("Hello ThinkRook Chatbot   :)")
st.caption("Welcome to your first chatbot app !!")

user_input = st.text_input("say somthing to the bot")

if user_input:
    st.write("You said:", user_input)
    st.write("The Bot says, Hello there!")

