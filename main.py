import streamlit as st
from utils import process_url_and_save_index, answer_query

st.set_page_config(page_title="Scheme Research Tool")
st.title("🔍 Scheme Research Tool")

st.sidebar.header("Step 1: Input URLs")
url_input = st.sidebar.text_area("Enter PDF/Article URLs (one per line):")

if st.sidebar.button("📥 Process URLs"):
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    if urls:
        with st.spinner("Processing and indexing URLs..."):
            process_url_and_save_index(urls)
        st.success("Documents processed and indexed!")
    else:
        st.warning("Please enter at least one URL.")

st.header("Step 2: Ask a question")
question = st.text_input("Ask a question about the schemes:")

if question:
    with st.spinner("Searching for answer..."):
        answer, sources = answer_query(question)
    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources")
        for src in sources:
            st.write(f"🔗 {src}")
