# We are using this lightweight app solely as a benchmark for our ChatSQC app implementation

# Example adapts https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app by changing
# only the model (line 21), temperature (lines 24-25), and temperature in line 51 to match our ChatSQC app


import openai
import streamlit as st
from dotenv import load_dotenv


# to load the API key for Open AI
load_dotenv()


st.title("A Quick ChatGPT-3.5 Benchmark Implementation")


# setting the open AI model to match what we have in ChatSQC
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"


if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.25 


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Please ask me questions about industrial statistics, quality, reliability, or experimental design?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            temperature=st.session_state["temperature"],
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
