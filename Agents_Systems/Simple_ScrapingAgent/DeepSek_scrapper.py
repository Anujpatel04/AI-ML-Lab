import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

env_path = Path("/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env")
load_dotenv(env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

st.title("Web Scrapping AI Agent 🕵️‍♂️")
st.caption("This app allows you to scrape a website using DeepSeek API")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ DEEPSEEK_API_KEY not found in .env file. Please add it to your .env file in the root directory.")
    st.stop()

model = st.radio(
    "Select the model",
    ["deepseek-chat", "deepseek-reasoner"],
    index=0,
)    

graph_config = {
    "llm": {
        "api_key": DEEPSEEK_API_KEY,
        "model": f"openai/{model}",  # scrapegraphai expects provider/model format
        "base_url": "https://api.deepseek.com",
    },
}

url = st.text_input("Enter the URL of the website you want to scrape")
user_prompt = st.text_input("What you want the AI agent to scrape from the website?")

if st.button("Scrape", disabled=not (url and user_prompt)):
    if url and user_prompt:
        with st.spinner("Scraping website..."):
            try:
                smart_scraper_graph = SmartScraperGraph(
                    prompt=user_prompt,
                    source=url,
                    config=graph_config
                )
                result = smart_scraper_graph.run()
                st.write(result)
            except Exception as e:
                st.error(f"Error during scraping: {str(e)}")
                st.exception(e)