from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class Paper(BaseModel):
    id: str
    title: str
    year: int
    abstract: str
    authors: List[str]
    topics: List[str]


class QueryResponse(BaseModel):
    answer: str
    papers: List[Paper]


class GraphStats(BaseModel):
    paper_count: int
    author_count: int
    topic_count: int
    written_by_count: int
    has_topic_count: int
