# PDF Chat Assistant

RAG application for querying PDF documents using DeepSeek API. Built with Streamlit and EmbedChain.

## Features

- PDF document upload and processing
- Interactive chat interface with message history
- DeepSeek AI integration for document Q&A
- Session-based conversation management
- Streamlit Cloud deployment support

## Requirements

- Python 3.8 or higher
- DeepSeek API key
- Dependencies listed in `requirements.txt`

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Set your DeepSeek API key using one of the following methods:

**Local Development:**
Create a `.env` file in the root directory:
```
DEEPSEEK_API_KEY=your-api-key-here
```

**Streamlit Cloud:**
Add to your app's secrets configuration:
```
DEEPSEEK_API_KEY = "your-api-key-here"
```

## Usage

1. Run the application:
   ```bash
   streamlit run chat_pdf.py
   ```

2. Upload a PDF document through the file uploader.

3. Wait for document processing to complete.

4. Enter questions in the chat interface to query the document.

5. View responses and maintain conversation history within the session.

## Deployment

### Streamlit Cloud

1. Push code to a GitHub repository.

2. Create a new app on Streamlit Cloud.

3. Configure the main file path: `All_LLM's/PDF_RAG/chat_pdf.py`

4. Add `DEEPSEEK_API_KEY` to app secrets.

5. Deploy the application.

## Application Structure

- `chat_pdf.py` - Main application file
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Notes

The `.env` file is shared across all projects in the root directory. For local development, ensure your API key is set in the root `.env` file.
