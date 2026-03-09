import asyncio
import os
import sys
import streamlit as st
from textwrap import dedent
from pathlib import Path
import warnings
import logging
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

logging.getLogger("mcp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*cancel scope.*")

try:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
except (NameError, AttributeError):
    script_dir = Path.cwd()
    root_dir = script_dir
    max_depth = 10
    depth = 0
    while depth < max_depth and root_dir != root_dir.parent:
        if (root_dir / "config.py").exists():
            break
        root_dir = root_dir.parent
        depth += 1

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

USE_SHARED_CONFIG = False
AZURE_KEY = None
AZURE_BASE_URL = None
API_VERSION = None
AZURE_MODEL = "gpt-4o"

try:
    config_path = root_dir / "config.py"
    if config_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        AZURE_KEY = getattr(config_module, "AZURE_KEY", None)
        AZURE_BASE_URL = getattr(config_module, "AZURE_BASE_URL", None)
        API_VERSION = getattr(config_module, "API_VERSION", None)
        AZURE_MODEL = getattr(config_module, "AZURE_MODEL", "gpt-4o")
        get_openai_client_config = getattr(config_module, "get_openai_client_config", None)
        
        if AZURE_KEY:
            USE_SHARED_CONFIG = True
except (ImportError, Exception) as e:
    USE_SHARED_CONFIG = False

st.set_page_config(page_title="GitHub MCP Agent", page_icon="", layout="wide")

st.markdown("<h1 class='main-header'>GitHub MCP Agent</h1>", unsafe_allow_html=True)
st.markdown("Explore GitHub repositories with natural language using the Model Context Protocol")

if USE_SHARED_CONFIG and AZURE_KEY:
    os.environ["OPENAI_API_KEY"] = AZURE_KEY
    openai_key = AZURE_KEY
else:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("Azure OpenAI not configured. Please check config.py or set OPENAI_API_KEY environment variable.")
        st.stop()

with st.sidebar:
    st.header("Authentication")
    
    if USE_SHARED_CONFIG and AZURE_KEY:
        st.success("Azure OpenAI configured from shared config")
    else:
        st.warning("Using OpenAI API key from environment variable")
    
    github_token = st.text_input("GitHub Token", type="password", 
                                help="Create a token with repo scope at github.com/settings/tokens")
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
    
    st.markdown("---")
    st.markdown("### System Status")
    import subprocess
    docker_available = False
    try:
        docker_check = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=3
        )
        docker_available = docker_check.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        docker_available = False
    
    if docker_available:
        st.success("Docker is running")
    else:
        st.error("Docker is not running. Please start Docker Desktop.")
        st.info("The GitHub MCP Agent requires Docker to run the GitHub MCP server.")
        with st.expander("How to start Docker Desktop"):
            st.markdown("""
            **On macOS:**
            1. Open Docker Desktop from Applications or Spotlight
            2. Wait for Docker to fully start (whale icon in menu bar should be steady)
            3. Refresh this page to check status again
            
            **Alternative:** Run `open -a Docker` in Terminal
            """)
    
    st.markdown("---")
    st.markdown("### Example Queries")
    
    st.markdown("**Issues**")
    st.markdown("- Show me issues by label")
    st.markdown("- What issues are being actively discussed?")
    
    st.markdown("**Pull Requests**")
    st.markdown("- What PRs need review?")
    st.markdown("- Show me recent merged PRs")
    
    st.markdown("**Repository**")
    st.markdown("- Show repository health metrics")
    st.markdown("- Show repository activity patterns")
    
    st.markdown("---")
    st.caption("Note: Always specify the repository in your query if not already selected in the main input.")

col1, col2 = st.columns([3, 1])
with col1:
    repo = st.text_input("Repository", value="Anujpatel04/Anuj-AI-ML-Lab", help="Format: owner/repo")
with col2:
    query_type = st.selectbox("Query Type", [
        "Issues", "Pull Requests", "Repository Activity", "Custom"
    ])

if query_type == "Issues":
    query_template = f"Find issues labeled as bugs in {repo}"
elif query_type == "Pull Requests":
    query_template = f"Show me recent merged PRs in {repo}"
elif query_type == "Repository Activity":
    query_template = f"Analyze code quality trends in {repo}"
else:
    query_template = ""

query = st.text_area("Your Query", value=query_template, 
                     placeholder="What would you like to know about this repository?")

async def run_github_agent(message):
    if not os.getenv("GITHUB_TOKEN"):
        return "Error: GitHub token not provided"
    
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OpenAI API key not provided"
    
    # Check if Docker is available
    import subprocess
    try:
        docker_check = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5
        )
        if docker_check.returncode != 0:
            return "Error: Docker is not running. Please start Docker Desktop and try again."
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "Error: Docker is not installed or not running. Please install Docker Desktop and ensure it's running."
    
    try:
        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run", "-i", "--rm",
                "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "-e", "GITHUB_TOOLSETS",
                "ghcr.io/github/github-mcp-server"
            ],
            env={
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_TOKEN'),
                "GITHUB_TOOLSETS": "repos,issues,pull_requests"
            }
        )
        
        import contextlib
        import sys
        from io import StringIO
        
        class FilteredStderr:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                self.buffer = StringIO()
                self.error_lines = []
                
            def write(self, text):
                self.error_lines.append(text)
                
                full_text = ''.join(self.error_lines[-10:])
                
                if any(keyword in full_text.lower() for keyword in [
                    "cancel scope", "different task", "async_generator", 
                    "stdio_client", "an error occurred during closing",
                    "exception group", "generatorexit", "runtimeerror: attempted to exit",
                    "attempted to exit cancel scope"
                ]):
                    self.error_lines = []
                    return len(text)
                
                if len(self.error_lines) > 20:
                    self.error_lines = self.error_lines[-10:]
                
                return self.original_stderr.write(text)
                
            def flush(self):
                self.original_stderr.flush()
                
            def __getattr__(self, name):
                return getattr(self.original_stderr, name)
        
        original_stderr = sys.stderr
        filtered_stderr = FilteredStderr(original_stderr)
        
        try:
            sys.stderr = filtered_stderr
        
        async with MCPTools(server_params=server_params) as mcp_tools:
                if USE_SHARED_CONFIG and AZURE_BASE_URL:
                    os.environ["OPENAI_API_KEY"] = AZURE_KEY
                    os.environ["OPENAI_BASE_URL"] = AZURE_BASE_URL
                    if "?" not in AZURE_BASE_URL:
                        azure_base_url_with_version = f"{AZURE_BASE_URL}?api-version={API_VERSION}"
                    else:
                        azure_base_url_with_version = AZURE_BASE_URL
                    os.environ["OPENAI_BASE_URL"] = azure_base_url_with_version
                
            agent = Agent(
                tools=[mcp_tools],
                    model=AZURE_MODEL if USE_SHARED_CONFIG else "gpt-4o",
                instructions=dedent("""\
                    You are a GitHub assistant. Help users explore repositories and their activity.
                    - Provide organized, concise insights about the repository
                    - Focus on facts and data from the GitHub API
                    - Use markdown formatting for better readability
                    - Present numerical data in tables when appropriate
                    - Include links to relevant GitHub pages when helpful
                """),
                markdown=True,
            )
            
            response: RunOutput = await asyncio.wait_for(agent.arun(message), timeout=120.0)
                result = response.content
        finally:
            sys.stderr = original_stderr
            await asyncio.sleep(0.2)
            sys.stderr = original_stderr
            await asyncio.sleep(0.1)
        
        return result
                
    except asyncio.TimeoutError:
        return "Error: Request timed out after 120 seconds"
    except subprocess.CalledProcessError as e:
        if "docker" in str(e).lower() or "daemon" in str(e).lower():
            return "Error: Docker is not running. Please start Docker Desktop and try again."
        return f"Error: {str(e)}"
    except RuntimeError as e:
        error_msg = str(e)
        if "cancel scope" in error_msg.lower() or "different task" in error_msg.lower():
            return "Warning: Connection cleanup issue occurred. Please try your query again."
        raise
    except Exception as e:
        error_msg = str(e)
        if "docker" in error_msg.lower() or "daemon" in error_msg.lower() or "Cannot connect" in error_msg:
            return "Error: Docker is not running. Please start Docker Desktop and try again."
        return f"Error: {str(e)}"

if st.button("Run Query", type="primary", use_container_width=True):
    if not openai_key:
        st.error("Azure OpenAI API key not configured. Please check config.py or set OPENAI_API_KEY environment variable.")
    elif not github_token:
        st.error("Please enter your GitHub token in the sidebar")
    elif not query:
        st.error("Please enter a query")
    else:
        with st.spinner("Analyzing GitHub repository..."):
            if repo and repo not in query:
                full_query = f"{query} in {repo}"
            else:
                full_query = query
                
            result = asyncio.run(run_github_agent(full_query))
        
        st.markdown("### Results")
        st.markdown(result)

if 'result' not in locals():
    st.markdown(
        """<div class='info-box'>
        <h4>How to use this app:</h4>
        <ol>
            <li>Enter your <strong>GitHub token</strong> in the sidebar</li>
            <li>Specify a repository (e.g., Anujpatel04/awesome-llm-apps)</li>
            <li>Select a query type or write your own</li>
            <li>Click 'Run Query' to see results</li>
        </ol>
        <p><strong>How it works:</strong></p>
        <ul>
            <li>Uses Azure OpenAI from shared config (configured automatically)</li>
            <li>Uses the official GitHub MCP server via Docker for real-time access to GitHub API</li>
            <li>AI Agent (powered by Azure OpenAI) interprets your queries and calls appropriate GitHub APIs</li>
            <li>Results are formatted in readable markdown with insights and links</li>
            <li>Queries work best when focused on specific aspects like issues, PRs, or repository info</li>
        </ul>
        </div>""", 
        unsafe_allow_html=True
    )
