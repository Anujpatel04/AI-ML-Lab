"""Ingestion pipelines for Adaptive RAG."""
from .vector_ingest import run as run_vector_ingest
from .graph_ingest import run as run_graph_ingest

__all__ = ["run_vector_ingest", "run_graph_ingest"]
