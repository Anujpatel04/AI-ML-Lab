"""
Adaptive AI Interview Simulator – FastAPI application entry point.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import LOG_LEVEL
from api.interview_routes import router

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

app = FastAPI(
    title="Adaptive AI Interview Simulator",
    description="Role-adaptive technical interviews from job description (LangChain + Azure OpenAI)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
def root():
    return {"service": "Adaptive AI Interview Simulator", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
