import os
import sys
from typing import Optional

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 3
EMBEDDING_DIMENSION = 1536

SYSTEM_PROMPT = """You are a precise document assistant. Your task is to answer questions based solely on the provided context from a PDF document.

Rules:
1. Only use information from the provided context to answer questions.
2. Always cite the page number(s) where you found the information.
3. If the answer is not found in the context, clearly state that the information is not available in the provided pages.
4. Do not make assumptions or add information beyond what is in the context.
5. Keep answers concise and factual.
6. Format page references as: (Page X) or (Pages X, Y, Z)."""

def extract_pages_from_pdf(pdf_file) -> list[dict]:
    """Extract text content from each page of the PDF with page metadata."""
    pages = []
    try:
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_number": page_num,
                    "content": text.strip()
                })
    except Exception as e:
        st.error(f"Failed to read PDF: {str(e)}")
        return []
    return pages


def create_embeddings(texts: list[str]) -> Optional[np.ndarray]:
    """Generate embeddings for a list of texts using OpenAI API."""
    if not texts:
        return None
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        st.error(f"Failed to create embeddings: {str(e)}")
        return None


def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.IndexFlatIP]:
    """Build a FAISS index from embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return None
    try:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Failed to build index: {str(e)}")
        return None


def retrieve_relevant_pages(
    query: str,
    index: faiss.IndexFlatIP,
    pages: list[dict],
    top_k: int = TOP_K
) -> list[dict]:
    """Retrieve the top-k most relevant pages for a query."""
    try:
        query_embedding = create_embeddings([query])
        if query_embedding is None:
            return []
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, min(top_k, len(pages)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(pages):
                results.append({
                    "page_number": pages[idx]["page_number"],
                    "content": pages[idx]["content"],
                    "score": float(score)
                })
        return results
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        return []


def generate_answer(question: str, retrieved_pages: list[dict]) -> str:
    """Generate an answer using the OpenAI Chat API based on retrieved context."""
    if not retrieved_pages:
        return "No relevant content found in the document to answer this question."
    
    context_parts = []
    for page in retrieved_pages:
        context_parts.append(f"[Page {page['page_number']}]\n{page['content']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""Context from the PDF document:

{context}

---

Question: {question}

Provide a concise answer based only on the context above. Reference the page numbers where you found the information."""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate answer: {str(e)}"


def format_source_pages(retrieved_pages: list[dict]) -> str:
    """Format the source page numbers for display."""
    if not retrieved_pages:
        return "None"
    page_numbers = sorted(set(p["page_number"] for p in retrieved_pages))
    return ", ".join(str(p) for p in page_numbers)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


def process_pdf(uploaded_file) -> bool:
    """Process an uploaded PDF file and build the vector index."""
    if uploaded_file is None:
        return False
    
    if st.session_state.pdf_name == uploaded_file.name:
        return True
    
    with st.spinner("Processing PDF..."):
        pages = extract_pages_from_pdf(uploaded_file)
        if not pages:
            st.error("No text content could be extracted from the PDF.")
            return False
        
        texts = [p["content"] for p in pages]
        embeddings = create_embeddings(texts)
        if embeddings is None:
            return False
        
        index = build_faiss_index(embeddings)
        if index is None:
            return False
        
        st.session_state.pages = pages
        st.session_state.index = index
        st.session_state.pdf_name = uploaded_file.name
        
    return True


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="PDF Question Answering",
        page_icon=None,
        layout="centered"
    )
    
    st.title("PDF Question Answering")
    st.markdown("Upload a PDF document and ask questions about its content.")
    st.divider()
    
    initialize_session_state()
    
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Select a PDF file to analyze"
    )
    
    if uploaded_file:
        if not process_pdf(uploaded_file):
            st.stop()
        
        st.success(f"Loaded: {uploaded_file.name} ({len(st.session_state.pages)} pages)")
        st.divider()
        
        question = st.text_input(
            "Question",
            placeholder="Enter your question about the document...",
            key="question_input"
        )
        
        if st.button("Submit", type="primary"):
            if not question or not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching document..."):
                    retrieved_pages = retrieve_relevant_pages(
                        question,
                        st.session_state.index,
                        st.session_state.pages,
                        TOP_K
                    )
                
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, retrieved_pages)
                
                st.divider()
                st.subheader("Answer")
                st.markdown(answer)
                
                st.divider()
                st.subheader("Sources")
                source_pages = format_source_pages(retrieved_pages)
                st.text(f"Referenced Pages: {source_pages}")
    else:
        st.info("Please upload a PDF document to begin.")


if __name__ == "__main__":
    main()
