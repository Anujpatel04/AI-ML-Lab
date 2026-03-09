import streamlit as st
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from manager import TourManager
from agents import set_default_openai_key
import json

# Load environment variables from root .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
load_dotenv()  # Also load from current directory if exists

def tts(text):
    from pathlib import Path
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please add OPENAI_API_KEY to .env file.")
    
    client = OpenAI(api_key=api_key)
    speech_file_path = Path(__file__).parent / f"speech_tour.mp3"
        
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def run_async(func, *args, **kwargs):
    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))

# Set page config for a better UI
st.set_page_config(
    page_title="AI Audio Tour Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load API key from environment
default_api_key = os.getenv("OPENAI_API_KEY", "")

# Sidebar for API key status
with st.sidebar:
    st.title("Settings")
    if default_api_key:
        st.success("OpenAI API key loaded from .env")
        api_key = default_api_key
    else:
        st.warning("OpenAI API key not found in .env")
        api_key = st.text_input("OpenAI API Key:", type="password", help="Enter your API key or add it to .env file")
        if api_key:
            st.session_state["OPENAI_API_KEY"] = api_key
            st.success("API key saved for this session!")

# Use session state API key if available, otherwise use env
api_key = st.session_state.get("OPENAI_API_KEY", default_api_key)
if api_key:
    set_default_openai_key(api_key)

# Main content
st.title("🎧 AI Audio Tour Agent")
st.markdown("""
    <div class='welcome-card'>
        <h3>Welcome to your personalized audio tour guide!</h3>
        <p>I'll help you explore any location with an engaging, natural-sounding tour tailored to your interests.</p>
    </div>
""", unsafe_allow_html=True)

# Create a clean layout with cards
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📍 Where would you like to explore?")
    location = st.text_input("Location", label_visibility="collapsed", placeholder="Enter a city, landmark, or location...")
    
    st.markdown("### 🎯 What interests you?")
    interests = st.multiselect(
        "Interests",
        label_visibility="collapsed",
        options=["History", "Architecture", "Culinary", "Culture"],
        default=["History", "Architecture"],
        help="Select the topics you'd like to learn about"
    )

with col2:
    st.markdown("### ⏱️ Tour Settings")
    duration = st.slider(
        "Tour Duration (minutes)",
        min_value=5,
        max_value=60,
        value=10,
        step=5,
        help="Choose how long you'd like your tour to be"
    )
    
    st.markdown("### 🎙️ Voice Settings")
    voice_style = st.selectbox(
        "Guide's Voice Style",
        options=["Friendly & Casual", "Professional & Detailed", "Enthusiastic & Energetic"],
        help="Select the personality of your tour guide"
    )

# Generate Tour Button
if st.button("🎧 Generate Tour", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar or add OPENAI_API_KEY to your .env file.")
    elif not location:
        st.error("Please enter a location.")
    elif not interests:
        st.error("Please select at least one interest.")
    else:
        with st.spinner(f"Creating your personalized tour of {location}..."):
            mgr = TourManager()
            final_tour = run_async(
                mgr.run, location, interests, duration
            )

            # Display the tour content in an expandable section
            with st.expander("📝 Tour Content", expanded=True):
                st.markdown(final_tour)
            
            # Add a progress bar for audio generation
            with st.spinner("🎙️ Generating audio tour..."):
                progress_bar = st.progress(0)
                tour_audio = tts(final_tour)
                progress_bar.progress(100)
            
            # Display audio player with custom styling
            st.markdown("### 🎧 Listen to Your Tour")
            st.audio(tour_audio, format="audio/mp3")
            
            # Add download button for the audio
            with open(tour_audio, "rb") as file:
                st.download_button(
                    label="📥 Download Audio Tour",
                    data=file,
                    file_name=f"{location.lower().replace(' ', '_')}_tour.mp3",
                    mime="audio/mp3"
                )