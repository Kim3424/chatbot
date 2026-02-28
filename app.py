import streamlit as st
from openai import OpenAI
import os

# Lấy API key từ environment (Streamlit Cloud / local)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Chatbot AI", page_icon="🤖")
st.title("🤖 Chatbot AI")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Bạn là chatbot AI nói tiếng Việt, giải thích dễ hiểu"}
    ]

for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
