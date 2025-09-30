import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_embedding_model():
    """Load a sentence-transformer model for embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_chromadb_collection(persist_directory="db"):
    """Initialize Chroma vector store (persistent)."""
    return Chroma(persist_directory=persist_directory, embedding_function=get_embedding_model())

def add_documents_to_chroma(chroma_db, docs):
    """Add text chunks to the Chroma vector store."""
    chroma_db.add_texts(docs)

def get_llama_pipeline(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """Load the Llama model using HuggingFace Transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def build_qa_chain(chroma_db):
    """Build a RetrievalQA chain using LangChain and Llama."""
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    llm = get_llama_pipeline()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain
