#!/usr/bin/env python3
"""Test document loading and chunking only (no API calls). Run from this dir: python test_load_chunk.py"""
from pathlib import Path
from utils.document_loader import load_documents, chunk_text

def main():
    data_dir = Path(__file__).resolve().parent / "data"
    print(f"Data dir: {data_dir}")
    print(f"Exists: {data_dir.is_dir()}")
    if not data_dir.is_dir():
        print("FAIL: data/ not found")
        return 1
    files = list(data_dir.rglob("*"))
    print(f"Files in data/: {[str(f.relative_to(data_dir)) for f in files if f.is_file()]}")
    docs = load_documents(data_dir)
    print(f"Docs loaded: {len(docs)}")
    if not docs:
        print("FAIL: no .txt/.md loaded")
        return 1
    for i, d in enumerate(docs):
        print(f"  Doc {i+1}: {len(d)} chars")
    chunk_size, overlap = 1000, 120
    chunks = chunk_text(docs, chunk_size=chunk_size, overlap=overlap)
    print(f"Chunks: {len(chunks)} (chunk_size={chunk_size}, overlap={overlap})")
    if not chunks:
        print("FAIL: no chunks produced")
        return 1
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}: {len(c)} chars, preview: {repr(c[:80])}...")
    if len(chunks) > 3:
        print(f"  ... and {len(chunks)-3} more")
    print("OK: load + chunk works")
    return 0

if __name__ == "__main__":
    exit(main())
