import os
import json
import tempfile
from pathlib import Path
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

try:
    import PyPDF2
except ImportError:
    st.error("Missing required package: PyPDF2. Please install it with: pip install PyPDF2")
    st.stop()

env_path = Path('/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env')
if not env_path.exists():
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

st.set_page_config(
    page_title="Resume Analysis & Job Suggestions",
    page_icon=None,
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .improvement-item {
        background-color: rgba(255, 152, 0, 0.12);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
        color: inherit;
    }
    .improvement-item strong {
        color: #e65100;
    }
    .job-suggestion {
        background-color: rgba(33, 150, 243, 0.12);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: inherit;
    }
    .job-suggestion strong {
        color: #1565c0;
    }
    [data-theme="dark"] .improvement-item {
        background-color: rgba(255, 152, 0, 0.25);
        border-left-color: #ffb74d;
    }
    [data-theme="dark"] .improvement-item strong {
        color: #ffb74d;
    }
    [data-theme="dark"] .job-suggestion {
        background-color: rgba(33, 150, 243, 0.25);
        border-left-color: #64b5f6;
    }
    [data-theme="dark"] .job-suggestion strong {
        color: #64b5f6;
    }
    .match-score {
        color: #1565c0;
    }
    [data-theme="dark"] .match-score {
        color: #64b5f6;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_file(file) -> str:
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        st.error(f"Unsupported file type: {file.type}")
        return ""

def analyze_resume_with_ai(resume_text: str) -> dict:
    if not DEEPSEEK_API_KEY:
        return {
            "error": "DEEPSEEK_API_KEY not found. Please add it to your .env file."
        }
    
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    
    prompt = f"""You are an expert resume reviewer and career advisor. Analyze the following resume and provide a comprehensive evaluation.

Resume Content:
{resume_text}

Please provide your analysis in the following JSON format:
{{
    "quality_score": <number between 0-100>,
    "quality_assessment": "<brief overall assessment>",
    "strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "improvements": [
        {{
            "category": "<category name>",
            "suggestion": "<specific improvement suggestion>",
            "priority": "<high/medium/low>"
        }}
    ],
    "job_suggestions": [
        {{
            "title": "<job title>",
            "match_score": <number 0-100>,
            "reason": "<why this job matches>"
        }}
    ],
    "skills_identified": ["<skill1>", "<skill2>", "<skill3>"],
    "experience_level": "<entry-level/mid-level/senior-level/executive>",
    "industry_suggestions": ["<industry1>", "<industry2>"]
}}

Focus on:
1. Resume structure, formatting, and clarity
2. Content quality (achievements, metrics, keywords)
3. Skills and experience alignment
4. ATS (Applicant Tracking System) compatibility
5. Job market alignment based on skills and experience

Provide specific, actionable feedback and realistic job suggestions based on the candidate's profile."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer and career advisor. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        return json.loads(result_text)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing AI response: {str(e)}")
        return {"error": "Failed to parse AI response"}
    except Exception as e:
        st.error(f"Error calling AI API: {str(e)}")
        return {"error": str(e)}

st.markdown('<h1 class="main-header">Resume Analysis & Job Suggestions</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your resume to get AI-powered feedback and job recommendations</p>', unsafe_allow_html=True)

if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found in .env file. Please add it to your .env file in the root directory.")
    st.stop()

st.divider()

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or TXT)",
    type=["pdf", "txt"],
    help="Upload your resume in PDF or text format"
)

if uploaded_file is not None:
    with st.spinner("Reading resume..."):
        resume_text = extract_text_from_file(uploaded_file)
    
    if resume_text:
        st.success(f"Resume loaded: {uploaded_file.name}")
        
        with st.expander("View Resume Content", expanded=False):
            st.text_area("Resume Text", resume_text, height=200, disabled=True)
        
        if st.button("Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume with AI... This may take a moment."):
                analysis = analyze_resume_with_ai(resume_text)
            
            if "error" in analysis:
                st.error(f"{analysis['error']}")
            else:
                st.success("Analysis complete!")
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Quality Score",
                        f"{analysis.get('quality_score', 0)}/100",
                        delta=None
                    )
                
                with col2:
                    experience = analysis.get('experience_level', 'Unknown')
                    st.metric("Experience Level", experience)
                
                with col3:
                    skills_count = len(analysis.get('skills_identified', []))
                    st.metric("Skills Identified", skills_count)
                
                st.divider()
                
                st.subheader("Quality Assessment")
                st.info(analysis.get('quality_assessment', 'No assessment available'))
                
                st.subheader("Strengths")
                strengths = analysis.get('strengths', [])
                if strengths:
                    for strength in strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.info("No specific strengths identified.")
                
                st.divider()
                
                st.subheader("Improvement Suggestions")
                improvements = analysis.get('improvements', [])
                if improvements:
                    for idx, improvement in enumerate(improvements, 1):
                        priority_label = {
                            "high": "[HIGH]",
                            "medium": "[MEDIUM]",
                            "low": "[LOW]"
                        }.get(improvement.get('priority', 'medium').lower(), "[MEDIUM]")
                        
                        st.markdown(f"""
                        <div class="improvement-item">
                            <strong>{priority_label} {improvement.get('category', 'General')}</strong><br>
                            {improvement.get('suggestion', '')}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific improvements suggested.")
                
                st.divider()
                
                st.subheader("Suggested Job Titles")
                job_suggestions = analysis.get('job_suggestions', [])
                if job_suggestions:
                    for job in job_suggestions:
                        match_score = job.get('match_score', 0)
                        st.markdown(f"""
                        <div class="job-suggestion">
                            <strong>{job.get('title', 'Unknown Position')}</strong> 
                            <span class="match-score" style="float: right; font-weight: bold;">Match: {match_score}%</span><br>
                            <em>{job.get('reason', 'No reason provided')}</em>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No job suggestions available.")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Skills Identified")
                    skills = analysis.get('skills_identified', [])
                    if skills:
                        st.write(", ".join(skills))
                    else:
                        st.info("No skills identified.")
                
                with col2:
                    st.subheader("Industry Suggestions")
                    industries = analysis.get('industry_suggestions', [])
                    if industries:
                        for industry in industries:
                            st.write(f"• {industry}")
                    else:
                        st.info("No industry suggestions available.")
    else:
        st.error("Could not extract text from the uploaded file. Please ensure the file is not corrupted.")

else:
    st.info("Please upload your resume to get started.")

