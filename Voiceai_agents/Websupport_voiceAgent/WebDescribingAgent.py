from typing import List, Dict, Optional
from pathlib import Path
import os
from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from agents import Agent, Runner
from openai import AsyncOpenAI
import tempfile
import uuid
from datetime import datetime
import time
import streamlit as st
from dotenv import load_dotenv
import asyncio

# Load environment variables from parent directory .env file
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
load_dotenv()  # Also load from current directory if exists

def init_session_state():
    defaults = {
        "initialized": False,
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "firecrawl_api_key": os.getenv("FIRECRAWL_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "doc_url": "",
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_config():
    with st.sidebar:
        st.title("🔑 Configuration")
        
        # Show status of API keys loaded from .env
        env_keys_loaded = []
        if st.session_state.qdrant_url:
            env_keys_loaded.append("Qdrant")
        if st.session_state.qdrant_api_key:
            env_keys_loaded.append("Qdrant API")
        if st.session_state.firecrawl_api_key:
            env_keys_loaded.append("Firecrawl")
        if st.session_state.openai_api_key:
            env_keys_loaded.append("OpenAI")
        
        if env_keys_loaded:
            st.success(f"✅ API Keys loaded from .env: {', '.join(env_keys_loaded)}")
        else:
            st.warning("⚠️ API keys not found in .env file. Please add them to your .env file.")
        
        st.markdown("---")
        st.markdown("### 📚 Documentation Setup")
        st.session_state.doc_url = st.text_input(
            "Documentation URL",
            value=st.session_state.doc_url,
            placeholder="https://docs.example.com",
            help="Enter the URL of the documentation website to crawl"
        )
        
        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
            help="Choose the voice for the audio response"
        )
        
        if st.button("Initialize System", type="primary"):
            # Check if all required API keys are loaded from .env
            missing_keys = []
            if not st.session_state.qdrant_url:
                missing_keys.append("QDRANT_URL")
            if not st.session_state.qdrant_api_key:
                missing_keys.append("QDRANT_API_KEY")
            if not st.session_state.firecrawl_api_key:
                missing_keys.append("FIRECRAWL_API_KEY")
            if not st.session_state.openai_api_key:
                missing_keys.append("OPENAI_API_KEY")
            
            if missing_keys:
                st.error(f"❌ Missing API keys in .env: {', '.join(missing_keys)}")
            elif not st.session_state.doc_url:
                st.error("❌ Please enter a Documentation URL")
            else:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    try:
                        st.markdown("🔄 Setting up Qdrant connection...")
                        client, embedding_model = setup_qdrant_collection(
                            st.session_state.qdrant_url,
                            st.session_state.qdrant_api_key
                        )
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        st.markdown("✅ Qdrant setup complete!")
                        
                        st.markdown("🔄 Crawling documentation pages...")
                        pages = crawl_documentation(
                            st.session_state.firecrawl_api_key,
                            st.session_state.doc_url
                        )
                        st.markdown(f"✅ Crawled {len(pages)} documentation pages!")
                        
                        store_embeddings(
                            client,
                            embedding_model,
                            pages,
                            "docs_embeddings"
                        )
                        
                        processor_agent, tts_agent = setup_agents(
                            st.session_state.openai_api_key
                        )
                        st.session_state.processor_agent = processor_agent
                        st.session_state.tts_agent = tts_agent
                        
                        st.session_state.setup_complete = True
                        st.success("✅ System initialized successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during setup: {str(e)}")

def setup_qdrant_collection(qdrant_url: str, qdrant_api_key: str, collection_name: str = "docs_embeddings"):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e
    
    return client, embedding_model

def crawl_documentation(firecrawl_api_key: str, url: str, output_dir: Optional[str] = None):
    firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
    pages = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use Firecrawl v2 API - scrape for single page (simpler and more reliable)
    try:
        # Scrape the URL with markdown and html formats
        response = firecrawl.scrape(url, formats=['markdown', 'html'])
        
        # Handle Document object from Firecrawl v2
        if hasattr(response, 'markdown'):
            content = response.markdown or getattr(response, 'html', '') or getattr(response, 'content', '')
            source_url = getattr(response, 'url', url) or url
            metadata = getattr(response, 'metadata', {}) or {}
        elif isinstance(response, dict):
            content = response.get('markdown') or response.get('html', '') or response.get('content', '')
            source_url = response.get('url', '') or url
            metadata = response.get('metadata', {})
        else:
            # Try to extract content from object attributes
            content = getattr(response, 'markdown', '') or getattr(response, 'html', '') or getattr(response, 'content', '')
            source_url = getattr(response, 'url', url) or url
            metadata = getattr(response, 'metadata', {}) or {}
        
        if content:
            if output_dir:
                filename = f"{uuid.uuid4()}.md"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Extract metadata fields
            if isinstance(metadata, dict):
                title = metadata.get('title', '')
                description = metadata.get('description', '')
                language = metadata.get('language', 'en')
            else:
                title = getattr(metadata, 'title', '') if metadata else ''
                description = getattr(metadata, 'description', '') if metadata else ''
                language = getattr(metadata, 'language', 'en') if metadata else 'en'
            
            pages.append({
                "content": content,
                "url": source_url,
                "metadata": {
                    "title": title,
                    "description": description,
                    "language": language,
                    "crawl_date": datetime.now().isoformat()
                }
            })
        else:
            raise Exception("No content extracted from the URL")
            
    except Exception as e:
        raise Exception(f"Failed to scrape URL: {str(e)}")
    
    return pages

def store_embeddings(client: QdrantClient, embedding_model: TextEmbedding, pages: List[Dict], collection_name: str):
    for page in pages:
        embedding = list(embedding_model.embed([page["content"]]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": page["content"],
                        "url": page["url"],
                        **page["metadata"]
                    }
                )
            ]
        )

def setup_agents(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""You are a helpful documentation assistant. Your task is to:
        1. Analyze the provided documentation content
        2. Answer the user's question clearly and concisely
        3. Include relevant examples when available
        4. Cite the source URLs when referencing specific content
        5. Keep responses natural and conversational
        6. Format your response in a way that's easy to speak out loud""",
        model="gpt-4o"
    )

    tts_agent = Agent(
        name="Text-to-Speech Agent",
        instructions="""You are a text-to-speech agent. Your task is to:
        1. Review the processed documentation response
        2. Optimize the text for speech (add natural pauses, emphasize important points)
        3. Handle technical terms clearly
        4. Keep the tone professional but friendly
        5. Ensure the text flows naturally when spoken
        6. Make minor adjustments to improve speech clarity""",
        model="gpt-4o-mini"
    )
    
    return processor_agent, tts_agent

async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    processor_agent: Agent,
    tts_agent: Agent,
    collection_name: str,
    openai_api_key: str
):
    try:
        query_embedding = list(embedding_model.embed([query]))[0]
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3,
            with_payload=True
        )
        
        search_results = search_response.points if hasattr(search_response, 'points') else []
        
        if not search_results:
            raise Exception("No relevant documents found in the vector database")
        
        context = "Based on the following documentation:\n\n"
        for result in search_results:
            payload = result.payload
            if not payload:
                continue
            url = payload.get('url', 'Unknown URL')
            content = payload.get('content', '')
            context += f"From {url}:\n{content}\n\n"
        
        context += f"\nUser Question: {query}\n\n"
        context += "Please provide a clear, concise answer that can be easily spoken out loud."
        
        processor_result = await Runner.run(processor_agent, context)
        processor_response = processor_result.final_output
        
        # Optimize text for speech using TTS agent
        tts_result = await Runner.run(tts_agent, processor_response)
        optimized_text = tts_result.final_output
        
        # Use OpenAI TTS API with correct model
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        audio_response = await async_openai.audio.speech.create(
            model="tts-1",  # Use tts-1 or tts-1-hd for high quality
            voice=st.session_state.selected_voice,
            input=optimized_text,
            response_format="mp3"
        )
        
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
        
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)
                
        return {
            "status": "success",
            "text_response": processor_response,
            "optimized_text": optimized_text,
            "audio_path": audio_path,
            "sources": [r.payload.get("url", "Unknown URL") for r in search_results if r.payload],
            "query_details": {
                "vector_size": len(query_embedding),
                "results_found": len(search_results),
                "collection_name": collection_name
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

def run_streamlit():
    st.set_page_config(
        page_title="Customer Support Voice Agent",
        page_icon="🎙️",
        layout="wide"
    )
    
    init_session_state()
    sidebar_config()
    
    st.title("🎙️ Customer Support Voice Agent")
    st.markdown("""
    Get OpenAI SDK voice-powered answers to your documentation questions! Simply:
    1. Configure your API keys in the sidebar
    2. Enter the documentation URL you want to learn about or have questions about
    3. Ask your question below and get both text and voice responses
    """)
    
    query = st.text_input(
        "What would you like to know about the documentation?",
        placeholder="e.g., How do I authenticate API requests?",
        disabled=not st.session_state.setup_complete
    )
    
    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                st.markdown("🔄 Searching documentation and generating response...")
                result = asyncio.run(process_query(
                    query,
                    st.session_state.client,
                    st.session_state.embedding_model,
                    st.session_state.processor_agent,
                    st.session_state.tts_agent,
                    "docs_embeddings",
                    st.session_state.openai_api_key
                ))
                
                if result["status"] == "success":
                    status.update(label="✅ Query processed!", state="complete")
                    
                    st.markdown("### Response:")
                    st.write(result["text_response"])
                    
                    if "audio_path" in result:
                        st.markdown(f"### 🔊 Audio Response (Voice: {st.session_state.selected_voice})")
                        st.audio(result["audio_path"], format="audio/mp3", start_time=0)
                        
                        with open(result["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.download_button(
                                label="📥 Download Audio Response",
                                data=audio_bytes,
                                file_name=f"voice_response_{st.session_state.selected_voice}.mp3",
                                mime="audio/mp3"
                            )
                    
                    st.markdown("### Sources:")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
                else:
                    status.update(label="❌ Error processing query", state="error")
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                    
            except Exception as e:
                status.update(label="❌ Error processing query", state="error")
                st.error(f"Error processing query: {str(e)}")
    
    elif not st.session_state.setup_complete:
        st.info("👈 Please configure the system using the sidebar first!")

if __name__ == "__main__":
    run_streamlit()