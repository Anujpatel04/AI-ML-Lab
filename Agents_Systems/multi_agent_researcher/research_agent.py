import os
from pathlib import Path

import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
from dotenv import load_dotenv


st.title("Multi-Agent AI Researcher")
st.caption("Research top HackerNews stories and users with a team of AI agents.")


root_env_path = Path(__file__).resolve().parents[2] / ".env"
if root_env_path.exists():
    load_dotenv(root_env_path)

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

if not openai_api_key:
    st.error(
        "OPENAI_API_KEY not found. Please add it to the root .env file at the repository root."
    )
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    hn_researcher = Agent(
        name="HackerNews Researcher",
        model=OpenAIChat(id="gpt-4o-mini"),
        role="Gets top stories from hackernews.",
        tools=[HackerNewsTools()],
    )

    web_searcher = Agent(
        name="Web Searcher",
        model=OpenAIChat(id="gpt-4o-mini"),
        role="Searches the web for information on a topic",
        tools=[DuckDuckGoTools()],
        add_datetime_to_context=True,
    )

    article_reader = Agent(
        name="Article Reader",
        model=OpenAIChat(id="gpt-4o-mini"),
        role="Reads articles from URLs.",
        tools=[Newspaper4kTools()],
    )

    hackernews_team = Team(
        name="HackerNews Team",
        model=OpenAIChat(id="gpt-4o-mini"),
        members=[hn_researcher, web_searcher, article_reader],
        instructions=[
            "First, search hackernews for what the user is asking about.",
            "Then, ask the article reader to read the links for the stories to get more information.",
            "Important: you must provide the article reader with the links to read.",
            "Then, ask the web searcher to search for each story to get more information.",
            "Finally, provide a thoughtful and engaging summary.",
        ],
        markdown=True,
        debug_mode=False,
        show_members_responses=False,
    )

    query = st.text_input("Enter your research query")

    if query:
        with st.spinner("Running agents..."):
            hn_response: RunOutput = hn_researcher.run(query, stream=False)
            web_response: RunOutput = web_searcher.run(query, stream=False)
            article_response: RunOutput = article_reader.run(query, stream=False)
            team_response: RunOutput = hackernews_team.run(query, stream=False)

        st.subheader("Team Summary")
        st.markdown(team_response.content)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### HackerNews Researcher")
            st.markdown(hn_response.content)

        with col2:
            st.markdown("### Web Searcher")
            st.markdown(web_response.content)

        with col3:
            st.markdown("### Article Reader")
            st.markdown(article_response.content)