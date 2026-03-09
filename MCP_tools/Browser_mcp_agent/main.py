import asyncio
import os
import sys
import streamlit as st
from textwrap import dedent
import yaml
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from config import get_azure_config, AZURE_KEY, AZURE_BASE_URL, API_VERSION, AZURE_MODEL
    USE_SHARED_CONFIG = True
except ImportError:
    USE_SHARED_CONFIG = False

_base_url = ""
_api_version = ""

if USE_SHARED_CONFIG:
    _base_url = AZURE_BASE_URL
    _api_version = API_VERSION
else:
    _config_file = Path("mcp_agent.config.yaml")
    if _config_file.exists():
        try:
            with open(_config_file, 'r') as f:
                _config = yaml.safe_load(f)
                _openai_config = _config.get("openai", {})
                _base_url = _openai_config.get("base_url", "")
                _api_version = _openai_config.get("api_version", "")
        except Exception:
            pass
            
if _base_url and "azure.com" in _base_url and _api_version:
    try:
        from openai import AsyncOpenAI
        _original_async_openai = AsyncOpenAI
        
        def _patched_async_openai(*args, **kwargs):
            if 'default_query' not in kwargs:
                kwargs['default_query'] = {'api-version': _api_version}
            elif 'api-version' not in kwargs.get('default_query', {}):
                if 'default_query' in kwargs:
                    kwargs['default_query']['api-version'] = _api_version
                else:
                    kwargs['default_query'] = {'api-version': _api_version}
            return _original_async_openai(*args, **kwargs)
        
        import mcp_agent.workflows.llm.augmented_llm_openai as _openai_module
        _openai_module.AsyncOpenAI = _patched_async_openai
    except Exception:
        pass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

def load_api_key_from_secrets():
    """Load API key from shared config, secrets file, or environment variables"""
    if os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"):
        return
    
    if USE_SHARED_CONFIG:
        try:
            os.environ["OPENAI_API_KEY"] = AZURE_KEY
            os.environ["AZURE_OPENAI_API_KEY"] = AZURE_KEY
            return
        except Exception:
            pass
    
    secrets_file = Path("mcp_agent.secrets.yaml")
    if secrets_file.exists():
        try:
            with open(secrets_file, 'r') as f:
                secrets = yaml.safe_load(f)
                if secrets and 'openai' in secrets and 'api_key' in secrets['openai']:
                    api_key = secrets['openai']['api_key']
                    if api_key and api_key != "YOUR_DEEPSEEK_API_KEY_HERE":
                        os.environ["OPENAI_API_KEY"] = api_key
                        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        except Exception as e:
            st.warning(f"Could not load secrets file: {e}")

load_api_key_from_secrets()

st.set_page_config(page_title="Browser MCP Agent", page_icon="🌐", layout="wide")

st.markdown("# 🌐 Browser MCP Agent")
st.markdown("### Control a web browser with natural language commands using AI-powered automation")

api_key_status = "✅ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Not Configured"
st.markdown(f"**API Key Status:** {api_key_status}")
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ Please configure your OpenAI API key in `mcp_agent.secrets.yaml`")

try:
    import yaml
    with open("mcp_agent.config.yaml", "r") as f:
        config = yaml.safe_load(f)
        openai_config = config.get("openai", {})
        base_url = openai_config.get("base_url", "")
        model = openai_config.get("default_model", "")
        
        if base_url and "deepseek.com" in base_url:
            st.error("""
            ⚠️ **DeepSeek API Compatibility Issue**
            
            DeepSeek API has a known compatibility issue with mcp-agent's message format. 
            Multi-step commands that use tool calls will fail with a JSON deserialization error.
            
            **Solutions:**
            1. **Use OpenAI API** (recommended) - Remove `base_url` from `mcp_agent.config.yaml` and use OpenAI API key
            2. **Use simple single-step commands** - Commands that complete in one turn may work
            3. **Wait for compatibility fix** - This may be resolved in future updates
            """)
except:
    pass

st.markdown("---")

with st.sidebar:
    st.markdown("## 🎯 Quick Examples")
    st.markdown("---")
    
    st.markdown("### 🌐 Navigation")
    example_nav = st.button("📌 Go to github.com/Anujpatel04/awesome-llm-apps", 
                           use_container_width=True, key="nav_example")
    if example_nav:
        st.session_state.example_query = "Go to github.com/Anujpatel04/awesome-llm-apps"
        st.rerun()
    
    st.markdown("### 🖱️ Interactions")
    example_interact = st.button("📌 Click on mcp_ai_agents", 
                                use_container_width=True, key="interact_example")
    if example_interact:
        st.session_state.example_query = "Click on mcp_ai_agents"
        st.rerun()
    
    example_scroll = st.button("📌 Scroll down to view more content", 
                               use_container_width=True, key="scroll_example")
    if example_scroll:
        st.session_state.example_query = "Scroll down to view more content"
        st.rerun()
    
    st.markdown("### 🔄 Multi-step Tasks")
    example_multi1 = st.button("📌 Navigate, scroll, and report details", 
                               use_container_width=True, key="multi1_example")
    if example_multi1:
        st.session_state.example_query = "Navigate to github.com/Anujpatel04/awesome-llm-apps, scroll down, and report details"
        st.rerun()
    
    example_multi2 = st.button("📌 Scroll and summarize readme", 
                               use_container_width=True, key="multi2_example")
    if example_multi2:
        st.session_state.example_query = "Scroll down and summarize the github readme"
        st.rerun()
    
    st.markdown("---")
    st.info("💡 **Note**: The agent uses Playwright to control a real browser. Click any example above to use it!")

# Query input
# Check if an example query was selected from sidebar
if 'example_query' in st.session_state and st.session_state.example_query:
    default_query = st.session_state.example_query
    st.session_state.example_query = None  # Clear after use
else:
    default_query = ""

query = st.text_area(
    "💬 Your Command", 
    value=default_query,
    placeholder="Enter a command like: 'Go to github.com' or 'Navigate to example.com and take a screenshot'",
    height=100,
    help="Type natural language commands to control the browser agent. See sidebar for examples."
)

# Initialize app and agent
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="streamlit_mcp_agent")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.is_processing = False
    st.session_state.example_query = None

async def setup_agent():
    if not st.session_state.initialized:
        try:
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()
            
            st.session_state.browser_agent = Agent(
                name="browser",
                instruction="""You are an autonomous web browsing agent. Your ONLY job is to execute browser commands using tools.

CRITICAL RULES:
1. NEVER respond with just text - ALWAYS use browser tools
2. When user says "go to X" or "navigate to X", IMMEDIATELY call playwright_browser_navigate with that URL
3. When user says "click X", first call playwright_browser_snapshot, then call playwright_browser_click
4. When user says "scroll", call playwright_browser_evaluate with scroll code
5. You MUST use tools for EVERY command - no exceptions

Tool usage examples:
- "Go to google.com" → playwright_browser_navigate(url="https://google.com")
- "Click login" → playwright_browser_snapshot() then playwright_browser_click()
- "Scroll down" → playwright_browser_evaluate(function="window.scrollBy(0, 500)")

After executing tools, briefly summarize what you did.""",
                server_names=["playwright"],
            )
            
            await st.session_state.browser_agent.initialize()
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            logger = st.session_state.mcp_agent_app.logger
            tools = await st.session_state.browser_agent.list_tools()
            logger.info("Tools available:", data=tools)
            
            st.session_state.initialized = True
        except Exception as e:
            return f"Error during initialization: {str(e)}"
    return None

async def run_mcp_agent(message):
    load_api_key_from_secrets()
    
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if deepseek_key and not openai_key:
        os.environ["OPENAI_API_KEY"] = deepseek_key
    
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: DeepSeek API key not provided. Please set DEEPSEEK_API_KEY environment variable or OPENAI_API_KEY, or configure it in mcp_agent.secrets.yaml"
    
    try:
        error = await setup_agent()
        if error:
            return error
        
        if message and message.strip():
            explicit_message = f"""EXECUTE THIS COMMAND NOW: {message}

You MUST use the available browser tools to execute this command. Do NOT just respond with text - actually perform the action using playwright_browser_navigate, playwright_browser_click, or other browser tools.

If the user wants to navigate somewhere, use playwright_browser_navigate immediately.
If the user wants to click something, first use playwright_browser_snapshot to see the page, then use playwright_browser_click.
If the user wants to scroll, use playwright_browser_evaluate with scroll JavaScript.

Execute the command now."""
        else:
            explicit_message = message
        
        result = await asyncio.wait_for(
            st.session_state.llm.generate_str(
                message=explicit_message, 
                request_params=RequestParams(use_history=False, maxTokens=4096)
            ),
            timeout=180.0
        )
        return result
    except asyncio.TimeoutError:
        return "❌ **Error: The request took too long to complete (timeout after 3 minutes).**\n\n**Possible causes:**\n- The website is slow or unresponsive\n- The task is too complex\n- Network connectivity issues\n\n**Try:**\n- Simplifying your command\n- Breaking it into smaller steps\n- Checking your internet connection"
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "authentication" in error_msg.lower():
            return f"❌ **Authentication Error:**\n\n```\n{error_msg}\n```\n\n**Solution:** Please check your API key in `mcp_agent.secrets.yaml`"
        elif "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower() or "429" in error_msg:
            return f"""❌ **API Quota Exceeded**

**The Issue:**
Your OpenAI API account has exceeded its current quota or doesn't have billing set up.

**Error Details:**
```
{error_msg}
```

**Solutions:**

1. **Add Credits to OpenAI Account:**
   - Visit: https://platform.openai.com/account/billing
   - Add payment method and credits
   - Check your usage limits

2. **Check Your Plan:**
   - Free tier has limited credits
   - Upgrade to a paid plan for more quota
   - Visit: https://platform.openai.com/account/usage

3. **Use DeepSeek API (Alternative):**
   - DeepSeek is more affordable but has compatibility limitations
   - Uncomment `base_url` in `mcp_agent.config.yaml`
   - Use your DeepSeek API key
   - Note: Multi-step commands may fail due to compatibility issues

4. **Get a New OpenAI API Key:**
   - If this is a shared/expired key, get a new one
   - Visit: https://platform.openai.com/api-keys"""
        elif "timeout" in error_msg.lower():
            return f"❌ **Timeout Error:**\n\n```\n{error_msg}\n```\n\n**Solution:** The request took too long. Try a simpler command or check your internet connection."
        elif "invalid type: sequence" in error_msg or "deserialize" in error_msg.lower():
            return f"""❌ **DeepSeek API Compatibility Error**

**The Issue:**
DeepSeek API has a compatibility issue with how mcp-agent formats tool call responses. The API expects message content as a string, but receives an array.

**Error Details:**
```
{error_msg}
```

**Possible Solutions:**

1. **Use OpenAI API instead** (recommended for now):
   - Change `base_url` in `mcp_agent.config.yaml` to remove it (uses OpenAI default)
   - Update your API key to an OpenAI key
   - This will work reliably with mcp-agent

2. **Try simpler commands** that don't require multiple tool calls:
   - Single-step navigation: "Go to example.com"
   - Simple actions that complete in one turn

3. **Wait for mcp-agent update** that fixes DeepSeek compatibility

**Note:** This is a known compatibility issue between mcp-agent and DeepSeek's API format. The first request works, but subsequent requests with tool results fail."""
        else:
            return f"❌ **Error:**\n\n```\n{error_msg}\n```\n\n**Troubleshooting:**\n- Check the logs for more details\n- Try a simpler command\n- Ensure all dependencies are installed"

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

def start_run():
    st.session_state.is_processing = True

col1, col2 = st.columns([3, 1])
with col1:
    st.button(
        "🚀 Run Command",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_processing,
        on_click=start_run,
    )
with col2:
    if st.button("🔄 Reset", use_container_width=True, help="Reset if stuck"):
        st.session_state.is_processing = False
        st.session_state.last_result = None
        st.rerun()

if st.session_state.is_processing:
    with st.spinner("Processing your request..."):
        try:
            result = st.session_state.loop.run_until_complete(
                asyncio.wait_for(run_mcp_agent(query), timeout=300.0)
            )
        except asyncio.TimeoutError:
            result = "❌ **Error: Request timed out after 5 minutes.**\n\nThis might happen if:\n- The website is slow to load\n- The agent is performing many steps\n- There's a network issue\n\n**Try:**\n- Simplifying your command\n- Checking your internet connection\n- Trying again with a shorter task"
        except Exception as e:
            result = f"❌ **Error occurred:**\n\n```\n{str(e)}\n```\n\n**Troubleshooting:**\n- Check if your DeepSeek API key is valid\n- Ensure you have internet connectivity\n- Try a simpler command first"
    st.session_state.last_result = result
    st.session_state.is_processing = False
    st.rerun()

if st.session_state.last_result:
    st.markdown("### 📋 Response")
    st.markdown(st.session_state.last_result)
    st.markdown("---")

if not st.session_state.last_result:
    st.markdown("## 📖 How to Use This App")
    
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Step 1: Configure API Key
        Your DeepSeek API key should already be configured in `mcp_agent.secrets.yaml`. 
        If you need to update it, edit that file.
        
        ### Step 2: Enter Your Command
        Type a natural language command in the text area above. Examples:
        - **Navigation**: "Go to github.com"
        - **Interaction**: "Click on the login button"
        - **Content**: "Summarize the main content of this page"
        - **Multi-step**: "Navigate to example.com, scroll down, and take a screenshot"
        
        ### Step 3: Run Command
        Click the **🚀 Run Command** button to execute your request.
        """)
    
    with st.expander("✨ App Capabilities", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🌐 Browser Navigation**
            - Visit any website
            - Navigate between pages
            - Go back/forward in history
            
            **🖱️ Interactions**
            - Click buttons and links
            - Fill out forms
            - Scroll through pages
            """)
        with col2:
            st.markdown("""
            **📸 Visual Features**
            - Take screenshots
            - Capture specific elements
            - Visual feedback
            
            **📊 Information Extraction**
            - Extract page content
            - Summarize information
            - Multi-step task completion
            """)
    
    with st.expander("💡 Example Commands", expanded=False):
        st.markdown("""
        **Basic Navigation:**
        ```
        Go to github.com/Anujpatel04/awesome-llm-apps
        Navigate to example.com
        Go back to the previous page
        ```
        
        **Interactions:**
        ```
        Click on the login button
        Scroll down to see more content
        Fill in the search box with "Python"
        ```
        
        **Content Extraction:**
        ```
        Summarize the main content of this page
        Extract the navigation menu items
        Take a screenshot of the hero section
        ```
        
        **Multi-step Tasks:**
        ```
        Go to github.com, find the most recent repository, and summarize it
        Navigate to example.com, scroll down, and report what you see
        Visit a news website and extract the top 3 headlines
        ```
        """)
    
    st.info("💡 **Tip**: Be specific in your commands for best results. The agent uses Playwright to control a real browser, so it can interact with websites just like you would!")

# Footer
st.markdown("---")
st.write("Built with Streamlit, Playwright, and [MCP-Agent](https://www.github.com/lastmile-ai/mcp-agent) Framework ❤️")
