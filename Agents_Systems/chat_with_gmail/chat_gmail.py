import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Check Python version compatibility with embedchain
if sys.version_info >= (3, 14):
    st.warning("Python 3.14+ detected. Embedchain requires Python <=3.13.")
    st.info("For best compatibility, use Python 3.13 or earlier:")
    st.code("""
python3.13 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
    """, language="bash")

# Monkey patch for chromadb compatibility with Pydantic v2
# This must be done BEFORE importing embedchain/chromadb
# We need to patch pydantic before any module tries to import BaseSettings from it
try:
    from pydantic_settings import BaseSettings
    import pydantic
    
    # Store original __getattr__ if it exists
    original_getattr = getattr(pydantic, '__getattr__', None)
    
    # Create a patched __getattr__ that returns BaseSettings when requested
    def patched_getattr(name):
        if name == 'BaseSettings':
            return BaseSettings
        if original_getattr:
            return original_getattr(name)
        raise AttributeError(f"module 'pydantic' has no attribute '{name}'")
    
    # Patch both the attribute and __getattr__
    pydantic.BaseSettings = BaseSettings
    pydantic.__getattr__ = patched_getattr
    
    # Also add to __all__ if it exists
    if hasattr(pydantic, '__all__'):
        if 'BaseSettings' not in pydantic.__all__:
            pydantic.__all__.append('BaseSettings')
    
    # Patch sys.modules to ensure the patched version is used
    sys.modules['pydantic'].BaseSettings = BaseSettings
    sys.modules['pydantic'].__getattr__ = patched_getattr
    
except (ImportError, Exception) as e:
    # If pydantic-settings is not available, try to import from pydantic directly
    try:
        from pydantic import BaseSettings
        import pydantic
        # Even if it exists, make sure it's accessible
        if not hasattr(pydantic, 'BaseSettings'):
            pydantic.BaseSettings = BaseSettings
    except ImportError:
        pass

# Set default environment variables for chromadb to avoid Pydantic v2 validation errors
# These are set before chromadb is imported
os.environ.setdefault('CHROMA_SERVER_HOST', 'localhost')
os.environ.setdefault('CHROMA_SERVER_HTTP_PORT', '8000')
os.environ.setdefault('CHROMA_SERVER_GRPC_PORT', '50051')
os.environ.setdefault('CLICKHOUSE_HOST', 'localhost')
os.environ.setdefault('CLICKHOUSE_PORT', '8123')

# Monkey patch for langchain compatibility
# langchain structure was changed in newer versions - many modules moved to langchain_community
try:
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    import types
    
    # Import langchain first to get the real module
    import langchain
    
    # Create docstore.document module if it doesn't exist
    if not hasattr(langchain, 'docstore'):
        langchain.docstore = types.ModuleType('langchain.docstore')
    
    if not hasattr(langchain.docstore, 'document'):
        langchain.docstore.document = types.ModuleType('langchain.docstore.document')
    
    # Add Document to the document module
    langchain.docstore.document.Document = Document
    
    # Create embeddings module if it doesn't exist
    if not hasattr(langchain, 'embeddings'):
        langchain.embeddings = types.ModuleType('langchain.embeddings')
    
    if not hasattr(langchain.embeddings, 'openai'):
        langchain.embeddings.openai = types.ModuleType('langchain.embeddings.openai')
    
    # Add OpenAIEmbeddings to the embeddings.openai module
    langchain.embeddings.openai.OpenAIEmbeddings = OpenAIEmbeddings
    
    # Create document_loaders module (moved to langchain_community in newer versions)
    if not hasattr(langchain, 'document_loaders'):
        langchain.document_loaders = types.ModuleType('langchain.document_loaders')
    
    # Try to import from langchain_community and add all loaders to langchain.document_loaders
    try:
        # Import specific loaders that embedchain needs
        from langchain_community.document_loaders import YoutubeLoader
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.document_loaders import WebBaseLoader
        
        # Add them to langchain.document_loaders
        langchain.document_loaders.YoutubeLoader = YoutubeLoader
        langchain.document_loaders.PyPDFLoader = PyPDFLoader
        langchain.document_loaders.WebBaseLoader = WebBaseLoader
        
        # Also try to import all other loaders dynamically
        import langchain_community.document_loaders as community_loaders
        for attr_name in dir(community_loaders):
            if not attr_name.startswith('_') and attr_name not in ['YoutubeLoader', 'PyPDFLoader', 'WebBaseLoader']:
                try:
                    attr = getattr(community_loaders, attr_name)
                    if not callable(attr) or (callable(attr) and hasattr(attr, '__module__') and 'document_loaders' in attr.__module__):
                        setattr(langchain.document_loaders, attr_name, attr)
                except (AttributeError, TypeError):
                    pass
    except ImportError:
        # If langchain_community is not available, try to import from langchain directly
        try:
            from langchain.document_loaders import YoutubeLoader, PyPDFLoader, WebBaseLoader
            langchain.document_loaders.YoutubeLoader = YoutubeLoader
            langchain.document_loaders.PyPDFLoader = PyPDFLoader
            langchain.document_loaders.WebBaseLoader = WebBaseLoader
        except ImportError:
            pass
    
    # Create text_splitter module (moved to langchain_text_splitters in newer versions)
    if not hasattr(langchain, 'text_splitter'):
        langchain.text_splitter = types.ModuleType('langchain.text_splitter')
    
    # Import RecursiveCharacterTextSplitter from langchain_text_splitters
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        langchain.text_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    except ImportError:
        # If langchain_text_splitters is not available, try to import from langchain directly
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            langchain.text_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        except ImportError:
            pass
    
    # Update sys.modules to ensure these are accessible
    sys.modules['langchain.docstore'] = langchain.docstore
    sys.modules['langchain.docstore.document'] = langchain.docstore.document
    sys.modules['langchain.embeddings'] = langchain.embeddings
    sys.modules['langchain.embeddings.openai'] = langchain.embeddings.openai
    sys.modules['langchain.document_loaders'] = langchain.document_loaders
    sys.modules['langchain.text_splitter'] = langchain.text_splitter
    
except (ImportError, AttributeError) as e:
    # If imports fail, try to import from langchain directly (older versions)
    try:
        from langchain.docstore.document import Document
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.document_loaders import YoutubeLoader
    except ImportError:
        pass

from embedchain import App

# Load .env from root directory
root_env_path = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env"
load_dotenv(root_env_path, override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
if DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY.strip()

def embedchain_bot(db_path, api_key):
    """Initialize embedchain app with DeepSeek API and ChromaDB"""
    config = {
        "llm": {
            "provider": "openai", 
            "config": {
                "model": "deepseek-chat", 
                "temperature": 0.5, 
                "api_key": api_key,
                "base_url": "https://api.deepseek.com"
            }
        },
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {
            "provider": "huggingface", 
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    }
    
    # Check if App.from_config exists
    if not hasattr(App, 'from_config'):
        st.error("Your embedchain version doesn't support `App.from_config()`. Embedchain requires Python <=3.13.")
        st.warning("Python 3.14 is not supported by embedchain. Please use Python 3.13 or earlier.")
        st.info("To fix this, create a new virtual environment with Python 3.13:")
        st.code("""
python3.13 -m venv myenv
source myenv/bin/activate
pip install embedchain[gmail]
        """, language="bash")
        return None
    
    try:
        # Use App.from_config (standard embedchain API)
        return App.from_config(config=config)
    except Exception as e:
        st.error(f"Error initializing embedchain: {e}")
        st.warning("Embedchain requires Python <=3.13. You are using Python 3.14.")
        st.info("Please use Python 3.13 or earlier to run this application.")
        return None

st.title("Chat with your Gmail Inbox")
st.caption("This app allows you to chat with your Gmail inbox using DeepSeek API")

if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    st.sidebar.success("DeepSeek API Key loaded successfully")
else:
    st.sidebar.error("API Key not found")

gmail_filter = "to: me label:inbox"

if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    if 'app' not in st.session_state:
        db_path = tempfile.mkdtemp()
        st.session_state.app = embedchain_bot(db_path, DEEPSEEK_API_KEY)
        st.session_state.emails_loaded = False
    
    # Check if app was initialized successfully
    if st.session_state.app is None:
        st.error("Failed to initialize embedchain app. Please check the error messages above.")
        st.stop()
    
    if not st.session_state.emails_loaded:
        with st.spinner("Loading emails from your Gmail inbox..."):
            try:
                st.session_state.app.add(gmail_filter, data_type="gmail")
                st.session_state.emails_loaded = True
                st.success(f"Added emails from Inbox to the knowledge base!")
            except Exception as e:
                st.error(f"Error loading emails: {e}")
                st.info("Make sure you have set up Gmail API credentials (credentials.json) in your working directory.")

    if st.session_state.emails_loaded:
        prompt = st.text_input("Ask any question about your emails")

        if prompt:
            try:
                with st.spinner("Thinking..."):
                    answer = st.session_state.app.chat(prompt)
                    st.write("**Answer:**")
                    st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.error("DeepSeek API key is not configured!")
    st.info("""
    **To set up your API key:**
    
    1. **Edit the .env file**: Open `.env` file in the root directory and set:
       ```
       DEEPSEEK_API_KEY=your-api-key-here
       ```
    
    2. **Or set environment variable**:
       ```bash
       export DEEPSEEK_API_KEY="your-api-key-here"
       ```
    
    **Note:** You also need to set up Gmail API credentials:
    - Go to [Google Cloud Console](https://console.cloud.google.com/)
    - Create a project and enable Gmail API
    - Create OAuth credentials and download as `credentials.json`
    - Place `credentials.json` in this directory
    """)
