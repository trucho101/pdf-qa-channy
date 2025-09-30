import streamlit as st
from utils import read_pdf, chunk_text
from backend import get_chromadb_collection, add_documents_to_chroma, build_qa_chain

st.set_page_config(page_title="PDF QA with Llama", layout="centered")
st.title("📄 PDF Document QA System with Llama")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.info("Extracting text from your PDF...", icon="📑")
    text = read_pdf(uploaded_file)
    chunks = chunk_text(text)
    st.success(f"PDF loaded and split into {len(chunks)} chunks.", icon="✅")

    chroma_db = get_chromadb_collection()
    add_documents_to_chroma(chroma_db, chunks)
    qa_chain = build_qa_chain(chroma_db)

    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."): 
            answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")