from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: int = Field(3, ge=1, le=10, description="How many chunks to retrieve")


class Source(BaseModel):
    text: str
    score: float
    document_title: Optional[str] = None
    organization: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    collection: str
    index_ready: bool
