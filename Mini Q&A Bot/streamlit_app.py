# streamlit_app.py
import streamlit as st
from rag.rag_pipeline import MiniQABot

bot = MiniQABot()

st.title("Mini Q&A Bot")
query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        answer = bot.answer(query)
        st.success(answer)
