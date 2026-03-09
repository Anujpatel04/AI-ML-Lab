import os
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.serpapi import SerpApiTools
from agno.tools.newspaper4k import Newspaper4kTools
import streamlit as st
from agno.models.openai import OpenAIChat

env_path = Path('/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env')

if not env_path.exists():
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

st.set_page_config(
    page_title="AI Journalist Agent",
    page_icon="🗞️",
    layout="wide"
)

st.title("AI Journalist Agent 🗞️")
st.caption("Generate High-quality articles with AI Journalist by researching, writing and editing quality articles on autopilot using DeepSeek API")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ DEEPSEEK_API_KEY not found in .env file. Please add it to your .env file in the root directory.")

if not SERP_API_KEY:
    st.error("⚠️ SERP_API_KEY not found in .env file. Please add it to your .env file in the root directory.")

if DEEPSEEK_API_KEY and SERP_API_KEY:
    os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    
    try:
        from agno.models.openai.chat import OpenAIChat as OpenAIChatClass
        original_format_message = OpenAIChatClass._format_message
        
        def patched_format_message(self, message):
            """Patch to convert developer role to system role for DeepSeek compatibility"""
            formatted = original_format_message(self, message)
            if isinstance(formatted, dict):
                if formatted.get('role') == 'developer':
                    formatted['role'] = 'system'
            elif hasattr(formatted, 'role') and formatted.role == 'developer':
                formatted.role = 'system'
            return formatted
        
        OpenAIChatClass._format_message = patched_format_message
    except Exception as e:
        st.warning(f"Could not patch message formatting: {e}")
    
    deepseek_model = OpenAIChat(
        id="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    searcher = Agent(
        name="Searcher",
        model=deepseek_model,
        instructions=dedent(
            """\
        You are a world-class journalist for the New York Times. Given a topic, generate a list of 3 search terms
        for writing an article on that topic. Then search the web for each term, analyse the results
        and return the 10 most relevant URLs.
        
        Instructions:
        - Given a topic, first generate a list of 3 search terms related to that topic.
        - For each search term, use `search_google` and analyze the results.
        - From the results of all searches, return the 10 most relevant URLs to the topic.
        - Remember: you are writing for the New York Times, so the quality of the sources is important.
        """
        ),
        tools=[SerpApiTools(api_key=SERP_API_KEY)],
        add_datetime_to_context=True,
    )
    writer = Agent(
        name="Writer",
        model=deepseek_model,
        instructions=dedent(
            """\
        You are a senior writer for the New York Times. Given a topic and a list of URLs,
        your goal is to write a high-quality NYT-worthy article on the topic.
        
        Instructions:
        - Given a topic and a list of URLs, first read the articles using `get_article_text`.
        - Then write a high-quality NYT-worthy article on the topic.
        - The article should be well-structured, informative, and engaging.
        - Ensure the length is at least as long as a NYT cover story -- at a minimum, 15 paragraphs.
        - Ensure you provide a nuanced and balanced opinion, quoting facts where possible.
        - Remember: you are writing for the New York Times, so the quality of the article is important.
        - Focus on clarity, coherence, and overall quality.
        - Never make up facts or plagiarize. Always provide proper attribution.
        """
        ),
        tools=[Newspaper4kTools()],
        add_datetime_to_context=True,
        markdown=True,
    )

    editor = Agent(
        name="Editor",
        model=deepseek_model,
        instructions=dedent(
            """\
        You are a senior NYT editor. Given a topic and a draft article, your goal is to edit and refine it to meet NYT standards.
        
        Instructions:
        - Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.
        - The article should be extremely articulate and well written.
        - Focus on clarity, coherence, and overall quality.
        - Ensure the article is engaging and informative.
        - Remember: you are the final gatekeeper before the article is published.
        """
        ),
        add_datetime_to_context=True,
        markdown=True,
    )

    query = st.text_input("What do you want the AI journalist to write an Article on?")

    if query:
        with st.spinner("Processing..."):
            with st.spinner("🔍 Searching for relevant sources..."):
                search_response: RunOutput = searcher.run(query, stream=False)
                urls_text = search_response.content
            
            with st.spinner("✍️ Writing the article..."):
                writer_input = f"Topic: {query}\n\nRelevant URLs and sources:\n{urls_text}\n\nPlease write a comprehensive NYT-worthy article on this topic."
                writer_response: RunOutput = writer.run(writer_input, stream=False)
                draft_article = writer_response.content
            
            with st.spinner("📝 Editing and refining the article..."):
                editor_input = f"Topic: {query}\n\nDraft Article:\n{draft_article}\n\nPlease edit, proofread, and refine this article to meet NYT standards."
                final_response: RunOutput = editor.run(editor_input, stream=False)
                st.write(final_response.content)