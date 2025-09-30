import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

st.logo("https://dme2wmiz2suov.cloudfront.net/Institution(8663)/Logo/4216689-ThinkRook_Logo.png", size="large")

st.sidebar.title("Bot Settings")

with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")


st.title("Create with ThinkRook")
st.caption("Create attractive images with Gemini on ThinkRook ImagenBot.")


if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    if not gemini_api_key:
        st.info("Please add your Google API key to continue.")
        st.stop()

    client = genai.Client(api_key=gemini_api_key)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            st.session_state.messages.append({"role": "assistant", "content": part.text})
            st.chat_message("assistant").write(part.text)

        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            # image.save('gemini-native-image.png')
            # image.show()
            st.image(image, caption="Generated Image")

            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

            st.download_button(
                label='Download Image',
                data=img_bytes,
                file_name='thinkrook_imagenbot_image.png'
            )