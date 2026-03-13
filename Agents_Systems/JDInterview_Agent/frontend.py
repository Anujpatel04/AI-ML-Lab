"""
Streamlit frontend for Adaptive AI Interview Simulator.
Run from project root: streamlit run frontend.py (or from JDInterview_Agent: streamlit run frontend.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from agents import parse_jd, generate_questions, evaluate_answer, generate_followup, generate_final_report
from chains import get_questions_for_session, run_interview_round, build_final_report
from config.settings import DEFAULT_QUESTION_LIMIT
from models.schemas import ParsedJD
from voice_utils import text_to_speech_bytes, transcribe_audio

st.set_page_config(page_title="AI Interview Simulator", page_icon="🎯", layout="centered")

# Session state
if "parsed_jd" not in st.session_state:
    st.session_state.parsed_jd = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "history" not in st.session_state:
    st.session_state.history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "report" not in st.session_state:
    st.session_state.report = None
if "use_voice" not in st.session_state:
    st.session_state.use_voice = False
if "voice_answer" not in st.session_state:
    st.session_state.voice_answer = None

st.title("Adaptive AI Interview Simulator")
st.caption("Paste a job description, get role-specific questions, and run an interview with AI evaluation.")

# --- Step 1: Parse JD ---
with st.expander("Step 1: Job description", expanded=st.session_state.parsed_jd is None):
    jd = st.text_area("Paste job description", height=180, placeholder="Paste the full job description here...")
    if st.button("Parse JD"):
        if not (jd or "").strip():
            st.warning("Enter a job description.")
        else:
            with st.spinner("Parsing..."):
                try:
                    parsed = parse_jd(jd.strip())
                    st.session_state.parsed_jd = parsed
                    st.session_state.questions = []
                    st.session_state.history = []
                    st.session_state.current_q = 0
                    st.session_state.report = None
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    if st.session_state.parsed_jd:
        p = st.session_state.parsed_jd
        st.success(f"**Role:** {p.role} | **Level:** {p.experience_level}")
        st.write("**Skills:**", ", ".join(p.skills[:15]) + ("..." if len(p.skills) > 15 else ""))
        st.write("**Topics:**", ", ".join(p.topics[:10]) + ("..." if len(p.topics) > 10 else ""))

# --- Step 2: Generate questions ---
if st.session_state.parsed_jd and not st.session_state.questions:
    with st.expander("Step 2: Generate questions", expanded=True):
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        if st.button("Generate questions"):
            with st.spinner("Generating..."):
                try:
                    qs = get_questions_for_session(st.session_state.parsed_jd, difficulty=difficulty)
                    st.session_state.questions = qs[:DEFAULT_QUESTION_LIMIT]
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# --- Step 3: Interview ---
if st.session_state.questions and st.session_state.report is None:
    n = len(st.session_state.questions)
    done = len(st.session_state.history)
    st.progress(done / n if n else 0)
    if done < n:
        st.write(f"Question **{done + 1}** of **{n}**")
    else:
        st.write("All questions answered. Generate your final report below.")

    if done < n:
        q = st.session_state.questions[done]
        st.subheader("Question")
        st.write(q)

        # Voice practice: play question aloud
        use_voice = st.checkbox("Use voice practice", value=st.session_state.use_voice, key="use_voice_cb")
        st.session_state.use_voice = use_voice
        if use_voice:
            if st.button("Play question aloud"):
                with st.spinner("Generating speech..."):
                    try:
                        audio_bytes = text_to_speech_bytes(q)
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.error(str(e))
            st.caption("Record your answer below; it will be transcribed and used when you submit.")

        answer = st.text_area("Your answer", height=120, key=f"ans_{done}")

        # Voice input: record and transcribe
        if use_voice:
            try:
                audio_in = st.audio_input("Record your answer", sample_rate=16000)
                if audio_in is not None:
                    data = audio_in.read()
                    if data:
                        with st.spinner("Transcribing..."):
                            try:
                                st.session_state.voice_answer = transcribe_audio(data, "recording.wav")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
            except Exception as e:
                st.warning("Microphone input requires Streamlit 1.50+. Upgrade with: pip install streamlit>=1.50")
                st.caption("You can still type your answer above.")
            if st.session_state.voice_answer:
                st.write("**Transcription:** ", st.session_state.voice_answer)

        answer_to_use = (st.session_state.voice_answer or "").strip() or (answer or "").strip()
        if st.button("Submit answer"):
            if not answer_to_use:
                st.warning("Type your answer or record it with voice.")
            else:
                with st.spinner("Evaluating..."):
                    try:
                        eval_result, followup = run_interview_round(q, answer_to_use)
                        st.session_state.voice_answer = None
                        st.session_state.history.append({
                            "question": q,
                            "answer": answer_to_use,
                            "score": eval_result.score,
                            "feedback": eval_result.feedback,
                            "strengths": eval_result.strengths,
                            "weaknesses": eval_result.weaknesses,
                            "followup": followup,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    else:
        if st.button("Generate final report"):
            with st.spinner("Generating report..."):
                try:
                    hist = [
                        {"question": h["question"], "answer": h["answer"], "score": h["score"], "feedback": h["feedback"]}
                        for h in st.session_state.history
                    ]
                    st.session_state.report = build_final_report(hist)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# --- Show history (evaluations) ---
if st.session_state.history:
    with st.expander("Your answers and feedback", expanded=False):
        for i, h in enumerate(st.session_state.history, 1):
            st.markdown(f"**Q{i}:** {h['question'][:80]}...")
            st.caption(f"Score: {h['score']}/10")
            st.write(h["feedback"])
            if h.get("followup"):
                st.caption("Follow-up: " + h["followup"])
            st.divider()

# --- Final report ---
if st.session_state.report:
    st.subheader("Final report")
    r = st.session_state.report
    st.metric("Overall", r.overall_score)
    st.write("**Technical strengths**")
    st.write(", ".join(r.technical_strengths) if r.technical_strengths else "—")
    st.write("**Knowledge gaps**")
    st.write(", ".join(r.knowledge_gaps) if r.knowledge_gaps else "—")
    st.write("**Recommended topics**")
    st.write(", ".join(r.recommended_topics) if r.recommended_topics else "—")
    st.write("**Detailed feedback**")
    st.write(r.detailed_feedback)
    if st.button("Start new interview"):
        st.session_state.parsed_jd = None
        st.session_state.questions = []
        st.session_state.history = []
        st.session_state.current_q = 0
        st.session_state.report = None
        st.rerun()
