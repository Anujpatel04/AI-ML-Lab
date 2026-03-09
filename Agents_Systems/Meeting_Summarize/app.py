"""
Meeting Notes → Action Agent
A Streamlit application that transforms meeting transcripts into actionable summaries,
action items, and professional follow-up emails using OpenAI.
"""

import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------

# Load environment variables from .env file (two levels up from this file)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional meeting assistant. Your task is to analyze meeting notes or transcripts and extract:
1. A concise, executive-level summary
2. Clear action items with owners and deadlines
3. A professional follow-up email

IMPORTANT RULES:
- If an owner is not explicitly mentioned for a task, use "Not specified"
- If a deadline is not explicitly mentioned, use "Not specified"
- Summary should be 2-4 sentences maximum
- Action items should be clear, specific, and actionable
- The follow-up email should:
  * Have a professional greeting
  * Reference the meeting briefly
  * List key action items
  * Be polite and professional
  * End with a proper sign-off

You MUST respond with ONLY valid JSON in this exact schema:
{
  "summary": "string",
  "action_items": [
    {
      "task": "string",
      "owner": "string",
      "deadline": "string"
    }
  ],
  "follow_up_email": "string"
}

Do NOT include any text before or after the JSON. Output ONLY the JSON object."""

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def validate_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and len(api_key) > 0


def process_meeting_notes(notes: str) -> dict:
    """
    Send meeting notes to OpenAI and get structured output.
    
    Args:
        notes: The raw meeting transcript or notes
        
    Returns:
        Dictionary containing summary, action_items, and follow_up_email
        
    Raises:
        ValueError: If the API response is not valid JSON
        Exception: For API errors
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the following meeting notes:\n\n{notes}"}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    # Extract the response content
    content = response.choices[0].message.content
    
    # Parse JSON response
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from API: {e}")
    
    # Validate required fields exist
    required_fields = ["summary", "action_items", "follow_up_email"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field in response: {field}")
    
    # Ensure action_items have required sub-fields with defaults
    for item in result.get("action_items", []):
        item.setdefault("task", "Not specified")
        item.setdefault("owner", "Not specified")
        item.setdefault("deadline", "Not specified")
    
    return result


def display_action_items(action_items: list) -> None:
    """Display action items in a formatted table."""
    if not action_items:
        st.info("No action items identified in the meeting notes.")
        return
    
    # Create table header
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.markdown("**Task**")
    with col2:
        st.markdown("**Owner**")
    with col3:
        st.markdown("**Deadline**")
    
    st.divider()
    
    # Display each action item
    for idx, item in enumerate(action_items, 1):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown(f"{idx}. {item.get('task', 'Not specified')}")
        with col2:
            st.markdown(item.get("owner", "Not specified"))
        with col3:
            st.markdown(item.get("deadline", "Not specified"))


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Meeting Notes → Action Agent",
        page_icon="📋",
        layout="centered"
    )
    
    # Header
    st.title("📋 Meeting Notes → Action Agent")
    st.markdown("Transform your meeting notes into actionable summaries, tasks, and follow-up emails.")
    st.divider()
    
    # Check API key configuration
    if not validate_api_key():
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    # Input section
    st.subheader("📝 Meeting Transcript")
    meeting_notes = st.text_area(
        label="Paste your meeting notes or transcript below:",
        height=200,
        placeholder="Example:\nWe discussed finalizing the API design. John will complete the backend by Friday.\nSarah will review the UI next week. Deployment timeline is still undecided."
    )
    
    # Generate button
    generate_button = st.button("🚀 Generate Actions", type="primary", use_container_width=True)
    
    # Process when button is clicked
    if generate_button:
        # Validate input
        if not meeting_notes or not meeting_notes.strip():
            st.warning("⚠️ Please enter meeting notes before generating actions.")
            st.stop()
        
        # Process with loading spinner
        with st.spinner("Analyzing meeting notes..."):
            try:
                result = process_meeting_notes(meeting_notes.strip())
                
                # Store result in session state
                st.session_state["result"] = result
                
            except ValueError as e:
                st.error(f"❌ Error parsing response: {str(e)}")
                st.stop()
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.stop()
    
    # Display results if available
    if "result" in st.session_state:
        result = st.session_state["result"]
        
        st.divider()
        
        # Summary Section
        st.subheader("📌 Summary")
        st.info(result.get("summary", "No summary available."))
        
        st.divider()
        
        # Action Items Section
        st.subheader("✅ Action Items")
        display_action_items(result.get("action_items", []))
        
        st.divider()
        
        # Follow-up Email Section
        st.subheader("📧 Follow-Up Email")
        email_content = result.get("follow_up_email", "No email generated.")
        
        # Display email in a code block for easy copying
        st.code(email_content, language=None)
        
        # Copy button hint
        st.caption("💡 Tip: Click the copy icon in the top-right corner of the email box to copy.")
        
        # Clear results button
        st.divider()
        if st.button("🔄 Clear & Start Over", use_container_width=True):
            del st.session_state["result"]
            st.rerun()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
