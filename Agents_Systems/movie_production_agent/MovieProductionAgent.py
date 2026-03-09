import os
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.tools.serpapi import SerpApiTools
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

SERP_API_KEY = "5caaea9b145768c68cb9485f3db9013b09394d60d4de0ed41b72d8f1e4b8213d"

st.title("AI Movie Production Agent 🎬")
st.caption("Bring your movie ideas to life with the teams of script writing and casting AI agents")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ DEEPSEEK_API_KEY not found in .env file. Please add it to your .env file in the root directory.")

if DEEPSEEK_API_KEY:
    try:
        from agno.models.openai.chat import OpenAIChat as OpenAIChatClass
        original_format_message = OpenAIChatClass._format_message
        
        def patched_format_message(self, message, *args, **kwargs):
            """Patch to convert developer role to system role for DeepSeek compatibility"""
            if args or kwargs:
                formatted = original_format_message(self, message, *args, **kwargs)
            else:
                formatted = original_format_message(self, message)
            
            if isinstance(formatted, dict):
                if formatted.get('role') == 'developer':
                    formatted['role'] = 'system'
            elif hasattr(formatted, 'role') and formatted.role == 'developer':
                formatted.role = 'system'
            return formatted
        
        OpenAIChatClass._format_message = patched_format_message
    except Exception:
        pass
    
    deepseek_model = OpenAIChat(
        id="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    script_writer = Agent(
        name="ScriptWriter",
        model=deepseek_model,
        description=dedent(
            """\
        You are an expert screenplay writer. Given a movie idea and genre, 
        develop a compelling script outline with character descriptions and key plot points.
        """
        ),
        instructions=[
            "Write a script outline with 3-5 main characters and key plot points.",
            "Outline the three-act structure and suggest 2-3 twists.",
            "Ensure the script aligns with the specified genre and target audience.",
        ],
    )

    casting_director = Agent(
        name="CastingDirector",
        model=deepseek_model,
        description=dedent(
            """\
        You are a talented casting director. Given a script outline and character descriptions,
        suggest suitable actors for the main roles, considering their past performances and current availability.
        """
        ),
        instructions=[
            "Suggest 2-3 actors for each main role.",
            "Check actors' current status using `search_google`.",
            "Provide a brief explanation for each casting suggestion.",
            "Consider diversity and representation in your casting choices.",
        ],
        tools=[SerpApiTools(api_key=SERP_API_KEY)],
    )

    movie_producer = Team(
        name="MovieProducer",
        model=deepseek_model,
        members=[script_writer, casting_director],
        description="Experienced movie producer overseeing script and casting.",
        instructions=[
            "Ask ScriptWriter for a script outline based on the movie idea.",
            "Pass the outline to CastingDirector for casting suggestions.",
            "Summarize the script outline and casting suggestions.",
            "Provide a concise movie concept overview.",
        ],
        markdown=True,
    )

    movie_idea = st.text_area("Describe your movie idea in a few sentences:")
    genre = st.selectbox("Select the movie genre:", 
                         ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller"])
    target_audience = st.selectbox("Select the target audience:", 
                                   ["General", "Children", "Teenagers", "Adults", "Mature"])
    estimated_runtime = st.slider("Estimated runtime (in minutes):", 60, 180, 120)

    if st.button("Develop Movie Concept"):
        with st.spinner("Developing movie concept..."):
            input_text = (
                f"Movie idea: {movie_idea}, Genre: {genre}, "
                f"Target audience: {target_audience}, Estimated runtime: {estimated_runtime} minutes"
            )
            response: RunOutput = movie_producer.run(input_text, stream=False)
            st.write(response.content)