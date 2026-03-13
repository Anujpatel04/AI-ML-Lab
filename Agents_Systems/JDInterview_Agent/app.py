"""
App entry point for uvicorn. Use: uvicorn main:app --reload
"""

# Import app from main so that "uvicorn main:app" and "uvicorn app:app" both work
from main import app

__all__ = ["app"]
