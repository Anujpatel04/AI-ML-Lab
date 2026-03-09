#!/usr/bin/env python3
import os
import tempfile
import streamlit as st
from embedchain import App
from dotenv import load_dotenv
from streamlit_chat import message

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        max-height: 500px;
        overflow-y: auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .uploaded-file {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

def get_api_key():
    """Get API key from Streamlit secrets (for Cloud) or .env file (for local)"""
    try:
        if hasattr(st, 'secrets') and st.secrets is not None:
            if 'DEEPSEEK_API_KEY' in st.secrets:
                key = st.secrets['DEEPSEEK_API_KEY']
                if isinstance(key, str):
                    return key.strip()
    except Exception:
        pass
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        env_path = os.path.join(root_dir, '.env')
        load_dotenv(env_path, override=True)
    except Exception:
        load_dotenv(override=True)
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    return api_key if api_key and api_key != "your-deepseek-api-key-here" else None

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'app' not in st.session_state:
    st.session_state.app = None
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = None

def embedchain_bot(db_path, api_key):
    """Initialize the EmbedChain app"""
    return App.from_config(
        config={
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
    )

def clear_chat():
    """Clear chat history"""
    st.session_state.messages = []

def reset_app():
    """Reset the app and clear all data"""
    st.session_state.app = None
    st.session_state.pdf_uploaded = False
    st.session_state.pdf_name = None
    st.session_state.messages = []
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        import shutil
        try:
            shutil.rmtree(st.session_state.db_path)
        except:
            pass
    st.session_state.db_path = None

DEEPSEEK_API_KEY = get_api_key()

st.markdown('<h1 class="main-header">📄 PDF Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your PDF documents using AI-powered chat</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    
    if DEEPSEEK_API_KEY:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Not Found")
        st.info("""
        **Configure your API key:**
        
        **For Streamlit Cloud:**
        - Go to Settings → Secrets
        - Add: `DEEPSEEK_API_KEY = "your-key-here"`
        
        **For Local:**
        - Add to `.env` file:
        ```
        DEEPSEEK_API_KEY=your-key-here
        ```
        """)
    
    st.divider()
    
    if st.session_state.pdf_uploaded:
        st.success(f"📄 PDF Loaded: {st.session_state.pdf_name}")
    else:
        st.info("📤 Upload a PDF to get started")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_app()
            st.rerun()
    
    st.divider()
    
    st.caption("💡 **Tip:** Upload a PDF and ask questions about its content")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ **API Key Required**")
    st.info("""
    Please configure your DeepSeek API key to use this application.
    
    **For Streamlit Cloud deployment:**
    1. Go to your app settings
    2. Navigate to "Secrets" tab
    3. Add your API key:
    ```
    DEEPSEEK_API_KEY = "your-api-key-here"
    ```
    
    **For local development:**
    Create a `.env` file in the root directory with:
    ```
    DEEPSEEK_API_KEY=your-api-key-here
    ```
    """)
else:
    if st.session_state.app is None:
        st.session_state.db_path = tempfile.mkdtemp()
        st.session_state.app = embedchain_bot(st.session_state.db_path, DEEPSEEK_API_KEY)
    
    st.subheader("📤 Upload PDF Document")
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start chatting about its content",
        label_visibility="collapsed"
    )
    
    if pdf_file is not None:
        if not st.session_state.pdf_uploaded or st.session_state.pdf_name != pdf_file.name:
            with st.spinner("📥 Processing PDF... This may take a moment."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(pdf_file.getvalue())
                        temp_path = f.name
                    
                    st.session_state.app.add(temp_path, data_type="pdf_file")
                    os.remove(temp_path)
                    
                    st.session_state.pdf_uploaded = True
                    st.session_state.pdf_name = pdf_file.name
                    st.success(f"✅ **{pdf_file.name}** has been successfully loaded!")
                    
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.pdf_uploaded = False
        
        if st.session_state.pdf_uploaded:
            st.markdown(f"""
            <div class="uploaded-file">
                <strong>📄 Current Document:</strong> {st.session_state.pdf_name}
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("💬 Chat with Your PDF")
    
    if not st.session_state.pdf_uploaded:
        st.info("👆 Please upload a PDF file first to start chatting")
    else:
        chat_container = st.container()
        with chat_container:
            if st.session_state.messages:
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "user":
                        message(msg["content"], is_user=True, key=f"user_{i}")
                    else:
                        message(msg["content"], is_user=False, key=f"assistant_{i}")
            else:
                st.info("👋 Hi! I'm ready to answer questions about your PDF. What would you like to know?")
        
        user_input = st.chat_input("Ask a question about your PDF...")
        
        if user_input and st.session_state.pdf_uploaded:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.app.chat(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(f"Error: {str(e)}")
            
            st.rerun()

st.markdown("---")
st.caption("Powered by DeepSeek AI | Built with Streamlit & EmbedChain")
