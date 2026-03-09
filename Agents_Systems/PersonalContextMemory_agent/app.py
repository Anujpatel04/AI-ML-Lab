import json
import os
from datetime import datetime, timezone

import streamlit as st

APP_TITLE = "Personal Context Memory Agent"
MEMORY_FILENAME = "memory.json"
MEMORY_SCHEMA = {
    "preferences": [],
    "writing_style": [],
    "past_decisions": [],
    "last_updated": None,
}


def get_memory_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MEMORY_FILENAME)


def load_memory() -> dict:
    path = get_memory_path()
    if not os.path.exists(path):
        return MEMORY_SCHEMA.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return MEMORY_SCHEMA.copy()

    if not isinstance(data, dict):
        return MEMORY_SCHEMA.copy()
    for key in MEMORY_SCHEMA:
        if key not in data:
            data[key] = MEMORY_SCHEMA[key] if key != "last_updated" else None
    for key in ("preferences", "writing_style", "past_decisions"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def save_memory(memory: dict) -> None:
    memory["last_updated"] = datetime.now(timezone.utc).isoformat()
    path = get_memory_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=True)
    except OSError:
        st.error("Failed to save memory.json. Check file permissions.")


def clear_memory() -> None:
    save_memory(MEMORY_SCHEMA.copy())


def call_llm(prompt: str) -> str:
    # Replace this stub with your preferred LLM provider.
    # Example: OpenAI, Ollama, Azure OpenAI, etc.
    # Keep system prompt + memory context + user prompt separated upstream.
    user_prompt = extract_user_prompt(prompt)
    return fallback_response(user_prompt)


def build_system_prompt(memory_context: str) -> str:
    return (
        "You are a helpful assistant that answers user questions accurately and clearly. "
        "You are often used to draft replies to messages and emails; provide drafts when asked. "
        "Follow any known writing style preferences.\n"
        f"{memory_context}"
    )


def build_memory_context(memory: dict, user_text: str) -> str:
    # Lightweight relevance filter: only include entries that share a keyword
    # with the current prompt. This keeps context tight and avoids noise.
    def relevant(items: list[str]) -> list[str]:
        if not user_text.strip():
            return []
        user_lower = user_text.lower()
        hits = []
        for item in items:
            if any(token in user_lower for token in item.lower().split()):
                hits.append(item)
        return hits

    preferences = relevant(memory.get("preferences", []))
    writing_style = memory.get("writing_style", [])
    past_decisions = relevant(memory.get("past_decisions", []))

    if not (preferences or writing_style or past_decisions):
        return "Known user context:\n- Preferences: (none)\n- Writing style: (none)\n- Past decisions: (none)\n"

    lines = [
        "Known user context:",
        f"- Preferences: {', '.join(preferences) if preferences else '(none)'}",
        f"- Writing style: {', '.join(writing_style) if writing_style else '(none)'}",
        f"- Past decisions: {', '.join(past_decisions) if past_decisions else '(none)'}",
    ]
    return "\n".join(lines) + "\n"


def extract_user_prompt(full_prompt: str) -> str:
    marker = "User:"
    if marker not in full_prompt:
        return full_prompt.strip()
    tail = full_prompt.split(marker, 1)[1]
    if "Assistant:" in tail:
        tail = tail.split("Assistant:", 1)[0]
    return tail.strip()


def normalize_sentence(text: str) -> str:
    return " ".join(text.strip().split())


def split_sentences(text: str) -> list[str]:
    raw = []
    for chunk in text.replace("\n", ". ").split("."):
        raw.append(chunk)
    out = []
    for item in raw:
        item = normalize_sentence(item)
        if item:
            out.append(item)
    return out


def fallback_response(user_prompt: str) -> str:
    if not user_prompt:
        return "How can I help you today?"

    lower = user_prompt.lower()
    if "data science" in lower:
        return (
            "Data science combines statistics, programming, and domain knowledge "
            "to extract insights from data. If you share your goal, I can outline "
            "a focused learning path or help draft a response related to it."
        )

    if any(phrase in lower for phrase in ("what should i reply", "what should i respond", "reply to", "respond to")):
        if "coffee" in lower or "coffe" in lower:
            return (
                "Here are a few polite options you can send:\n\n"
                "1) “Hey! Thanks for asking. I’m tied up today and won’t be able to make it. "
                "Can we plan for another day?”\n\n"
                "2) “Appreciate the invite! I’m juggling a few things today, so I’ll pass for now. "
                "Let’s do it another time.”\n\n"
                "3) “Thanks for the invite! I can’t make it today, but I’d love to catch up soon. "
                "How about later this week?”"
            )
        return (
            "Here is a polite reply you can send:\n\n"
            "“Thanks for the message! I won’t be able to make it this time, "
            "but I appreciate the invite. Let’s connect soon.”"
        )

    if any(token in lower for token in ("email", "emails", "mail", "mails", "message", "messages")):
        return (
            "Share the exact message and your intent, and I will draft a reply "
            "in your preferred style."
        )

    return (
        "Got it. Tell me the key points and your preferred tone, and I will draft "
        "a response for you."
    )


def should_save_preference(text: str) -> list[str]:
    cues = [
        "i prefer",
        "i like",
        "i dislike",
        "i want",
        "i don't want",
        "i do not want",
        "my preference is",
        "prefer",
    ]
    findings = []
    for sentence in split_sentences(text):
        sentence_lower = sentence.lower()
        if any(cue in sentence_lower for cue in cues):
            if len(sentence) <= 140:
                findings.append(sentence)
    return findings


def detect_writing_style(text: str) -> list[str]:
    text_lower = text.lower()
    styles = []
    if "formal" in text_lower:
        styles.append("formal")
    if "casual" in text_lower:
        styles.append("casual")
    if "bullet" in text_lower or "bullets" in text_lower:
        styles.append("bullet-heavy")
    if "academic" in text_lower:
        styles.append("academic")
    if "concise" in text_lower or "short" in text_lower:
        styles.append("concise")
    if "detailed" in text_lower or "in-depth" in text_lower:
        styles.append("detailed")
    return styles


def detect_style_feedback(text: str) -> list[str]:
    text_lower = text.lower()
    styles = []
    if "more formal" in text_lower or "too casual" in text_lower:
        styles.append("formal")
    if "more casual" in text_lower or "too formal" in text_lower:
        styles.append("casual")
    if "use bullets" in text_lower or "bullet points" in text_lower:
        styles.append("bullet-heavy")
    if "shorter" in text_lower or "more concise" in text_lower:
        styles.append("concise")
    if "more detail" in text_lower or "more detailed" in text_lower:
        styles.append("detailed")
    return styles


def detect_usage_context(text: str) -> list[str]:
    text_lower = text.lower()
    if any(token in text_lower for token in ("email", "emails", "mail", "mails", "message", "messages")):
        if any(phrase in text_lower for phrase in ("use it to", "use this to", "mostly use", "i will use", "i'll use")):
            return ["Primary use: draft replies to messages and emails."]
    return []


def detect_decisions(text: str) -> list[str]:
    cues = [
        "we decided",
        "i decided",
        "we will",
        "i will",
        "we are going to",
        "i am going to",
        "final decision",
        "i choose",
        "we chose",
        "let's go with",
        "go with",
        "i don't want to",
        "i do not want to",
    ]
    decisions = []
    for sentence in split_sentences(text):
        sentence_lower = sentence.lower()
        if any(cue in sentence_lower for cue in cues):
            if len(sentence) <= 140:
                decisions.append(sentence)
    return decisions


def update_memory(memory: dict, user_text: str, agent_text: str) -> dict:
    additions = {
        "preferences": [],
        "writing_style": [],
        "past_decisions": [],
    }

    # User input is the primary source; agent response can also confirm a decision.
    additions["preferences"].extend(should_save_preference(user_text))
    additions["preferences"].extend(detect_usage_context(user_text))
    additions["writing_style"].extend(detect_writing_style(user_text))
    additions["writing_style"].extend(detect_style_feedback(user_text))
    additions["past_decisions"].extend(detect_decisions(user_text))

    if "confirmed" in agent_text.lower() or "decision" in agent_text.lower():
        additions["past_decisions"].extend(detect_decisions(agent_text))

    for key in ("preferences", "writing_style", "past_decisions"):
        existing = set(memory.get(key, []))
        for item in additions[key]:
            if item and item not in existing:
                memory[key].append(item)
                existing.add(item)

    return memory


def render_memory(memory: dict) -> None:
    st.subheader("Memory Viewer")
    st.json(memory)


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)

    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    memory = load_memory()

    st.subheader("User Input")
    user_input = st.text_area("Enter your prompt", height=150)

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button("Run Agent", type="primary")
    with col_clear:
        clear_clicked = st.button("Clear Memory")

    if clear_clicked:
        if st.session_state.get("confirm_clear", False):
            clear_memory()
            st.session_state.confirm_clear = False
            st.success("Memory cleared.")
        else:
            st.session_state.confirm_clear = True
            st.warning("Click 'Clear Memory' again to confirm.")

    if run_clicked:
        memory_context = build_memory_context(memory, user_input)
        system_prompt = build_system_prompt(memory_context)
        user_prompt = user_input.strip()
        prompt = f"{system_prompt}\nUser: {user_prompt}\nAssistant:"
        response = call_llm(prompt)
        st.session_state.last_response = response

        memory = update_memory(memory, user_prompt, response)
        save_memory(memory)
        st.success("Agent run complete.")

    render_memory(load_memory())

    st.subheader("Agent Response")
    st.write(st.session_state.last_response or "No response yet.")


if __name__ == "__main__":
    render_app()
