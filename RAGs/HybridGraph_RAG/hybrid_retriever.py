"""
Combine graph-based context (Neo4j) and vector-based context (Pinecone) into a single fused context.
"""

import logging
from typing import List, Optional

import config
from entity_extractor import extract_entities
from graph_store import query_graph
from vector_store import retrieve_documents

logger = logging.getLogger(__name__)


def hybrid_retrieve(
    query: str,
    *,
    client,
    chat_model: str,
    embedding_model: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    pinecone_index,
    top_k_graph: Optional[int] = None,
    top_k_vector: Optional[int] = None,
    vector_namespace: str = "documents",
    progress_callback=None,
) -> str:
    """
    Run entity extraction → graph search → vector retrieval and return fused context string.

    Args:
        query: User question.
        client: OpenAI/Azure client (for entity extraction and embeddings).
        chat_model: Model for entity extraction.
        embedding_model: Model for vector embeddings.
        neo4j_uri, neo4j_username, neo4j_password: Neo4j connection.
        pinecone_index: Pinecone index object.
        top_k_graph: Max graph relationship records (default from config).
        top_k_vector: Max vector chunks (default from config).
        vector_namespace: Pinecone namespace for documents.

    Returns:
        Single string: "Graph context:\n...\n\nDocument context:\n..." for the LLM.
    """
    query = (query or "").strip()
    if not query:
        return "No query provided."

    top_k_graph = top_k_graph if top_k_graph is not None else config.TOP_K_GRAPH
    top_k_vector = top_k_vector if top_k_vector is not None else config.TOP_K_VECTOR

    def _progress(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    # 1. Entity extraction
    _progress("Extracting entities...")
    entities = extract_entities(client, chat_model, query)
    if entities:
        logger.info("Extracted entities: %s", entities[:10])

    # 2. Graph search
    _progress("Searching graph (Neo4j)...")
    graph_ctx = query_graph(
        neo4j_uri,
        neo4j_username,
        neo4j_password,
        entities,
        limit=top_k_graph,
    )

    # 3. Vector retrieval
    _progress("Searching vectors (Pinecone)...")
    vector_chunks = retrieve_documents(
        pinecone_index,
        client,
        embedding_model,
        query,
        namespace=vector_namespace,
        top_k=top_k_vector,
    )
    doc_ctx = "\n\n".join(vector_chunks) if vector_chunks else "(No matching documents.)"

    # 4. Context fusion
    parts = []
    if graph_ctx.strip():
        parts.append(graph_ctx.strip())
    else:
        parts.append("Knowledge graph: (No graph data for this query.)")
    parts.append("Relevant documents (semantic search):\n" + doc_ctx)
    return "\n\n---\n\n".join(parts)
