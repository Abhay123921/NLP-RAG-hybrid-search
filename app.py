import streamlit as st
import requests

st.title("🧠 Intelligent Search System")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        res = requests.get(f"http://127.0.0.1:8000/smart_search?query={query}")
        data = res.json()

        st.subheader("🔍 System Info")
        st.write(f"Intent: {data['intent']}")
        st.write(f"Mode: {data['mode']}")
        st.write(f"Latency: {data['latency_ms']} ms")

        if "answer" in data:
            st.subheader("🤖 Answer")
            st.write(data["answer"])

        st.subheader("📄 Results")
        for r in data["results"]:
            st.write(r["text"][:200])
            st.write("---")