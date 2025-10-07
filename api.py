from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import generate_answer, retrieve


class AskPayload(BaseModel):
    question: str


app = FastAPI(title="Local RAG API", version="0.1.0")


@app.post("/ask")
def ask(payload: AskPayload) -> Dict[str, Any]:
    faiss_dir = os.getenv("FAISS_DIR", ".faiss_index")
    if not Path(faiss_dir).exists():
        raise HTTPException(
            status_code=400,
            detail="Index introuvable. Construisez-le d'abord via CLI.",
        )

    docs = retrieve(payload.question, faiss_dir)
    if not docs:
        return {"answer": "Aucun passage pertinent trouvé.", "sources": []}

    answer, sources = generate_answer(payload.question, docs)
    return {"answer": answer, "sources": sources}
