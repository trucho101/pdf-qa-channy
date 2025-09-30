import PyPDF2

def read_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for embedding.
    chunk_size: Number of words per chunk.
    overlap: Number of words to overlap between chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
