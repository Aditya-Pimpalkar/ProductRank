"""GET /v1/products/{id} — document detail (ARCHITECTURE §4)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from productrank.db import session_for
from productrank.models import Document
from productrank.schemas import Dataset, ProductResponse

router = APIRouter(prefix="/v1", tags=["products"])


@router.get("/products/{doc_id}", response_model=ProductResponse)
def get_product(doc_id: str, dataset: Dataset = Dataset.MSMARCO) -> ProductResponse:
    with session_for(dataset.value) as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
        return ProductResponse(id=doc.id, title=doc.title, text=doc.text, metadata=doc.doc_metadata)
