"""GET /v1/products/{id} — document detail (ARCHITECTURE §4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from productrank.db import get_session
from productrank.models import Document
from productrank.schemas import ProductResponse

router = APIRouter(prefix="/v1", tags=["products"])


@router.get("/products/{doc_id}", response_model=ProductResponse)
def get_product(doc_id: str, session: Session = Depends(get_session)) -> ProductResponse:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
    return ProductResponse(
        id=doc.id, title=doc.title, text=doc.text, metadata=doc.doc_metadata
    )
