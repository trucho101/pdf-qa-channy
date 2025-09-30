# PDF Document-Based Question Answering System

This project is a professional template for a PDF Document QA chatbot using LangChain, Llama, and ChromaDB. It allows you to upload academic PDFs and ask natural language questions about their content through a simple web interface.

## Features

- **PDF Ingestion:** Extracts and preprocesses text from uploaded PDF files.
- **Chunking:** Splits documents into overlapping text chunks for efficient retrieval.
- **Embeddings & Vector Store:** Creates embeddings and stores them in ChromaDB for semantic search.
- **Question Answering:** Uses Llama (via HuggingFace) with LangChain to generate accurate answers.
- **User Interface:** Built with Streamlit for easy interaction.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/trucho101/pdf-qa-channy.git
cd pdf-qa-channy
```

### 2. Install dependencies

It is recommended to use a Python 3.9+ environment.

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Usage

- Upload an academic PDF file using the web interface.
- Enter your question in the input box.
- Get answers extracted from your document!

## Notes

- The default Llama model (`meta-llama/Llama-2-7b-chat-hf`) may require a GPU with at least 16GB RAM. For smaller hardware, use a smaller model in `backend.py`.
- You can customize chunk size and overlap in `utils.py`.
- For production or multiple users, consider improving session management and persistent storage.

## Structure

```
pdf-qa-channy/
├── app.py
├── backend.py
├── utils.py
├── requirements.txt
└── README.md
```

---

Created by [trucho101](https://github.com/trucho101)