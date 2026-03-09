# Page-Indexed PDF RAG

A production-quality Retrieval Augmented Generation (RAG) application for querying PDF documents with page-level precision. Built with Streamlit, OpenAI, and FAISS.

## Features

- **Page-Based Retrieval**: Each PDF page is treated as an individual retrieval unit, preserving document structure
- **Source Attribution**: All answers include explicit page number references
- **Grounded Responses**: LLM answers are strictly based on retrieved content with no hallucination
- **In-Memory Vector Search**: Fast similarity search using FAISS
- **OpenAI Integration**: Uses text-embedding-3-small for embeddings and GPT-4o for generation

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd RAGs/PageIndexed_RAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the parent directory (`RAGs/.env`) with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the application:

```bash
streamlit run app.py
```

Then:

1. Upload a PDF document using the file uploader
2. Enter your question in the text input field
3. Click Submit to get an answer with source page references

## Project Structure

```
PageIndexed_RAG/
├── app.py              # Main application (single-file implementation)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## How It Works

1. **PDF Processing**: The uploaded PDF is parsed page-by-page using pypdf. Each page's text content is extracted and stored with its page number.
2. **Embedding Creation**: Page contents are converted to vector embeddings using OpenAI's embedding API.
3. **Index Building**: Embeddings are normalized and added to a FAISS inner-product index for fast similarity search.
4. **Query Processing**: User questions are embedded and compared against the page index to find the most relevant pages.
5. **Answer Generation**: Retrieved page content is passed to GPT-4o with a strict system prompt that enforces grounded, citation-backed responses.
