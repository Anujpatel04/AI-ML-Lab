import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph

st.title("Web Scrapping AI Agent 🕵️‍♂️")
st.caption("This app allows you to scrape a website using Llama 3.1")

graph_config = {
    "llm": {
        "model": "ollama/llama3.1:latest",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "verbose": True,
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
