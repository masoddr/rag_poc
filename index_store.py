from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Set

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def _get_embeddings_model() -> HuggingFaceEmbeddings:
    model_name = os.getenv(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    logger.info("Using embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def build_faiss_index(documents: List[Document], faiss_dir: str) -> None:
    """Create a FAISS index on disk from provided documents.

    Overwrites any existing index at the given path.
    """
    embeddings = _get_embeddings_model()
    vectorstore = FAISS.from_documents(documents, embeddings)
    Path(faiss_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(faiss_dir)
    logger.info("FAISS index built at %s", faiss_dir)


def upsert_faiss_index(documents: List[Document], faiss_dir: str) -> None:
    """Create or update an existing FAISS index with new documents."""
    if Path(faiss_dir).exists():
        vectorstore = load_faiss_index(faiss_dir)
        # Simple dedup by content hash to avoid duplicates
        existing_texts: Set[str] = set()
        for doc in vectorstore.docstore._dict.values():  # type: ignore[attr-defined]
            existing_texts.add(doc.page_content)

        new_docs = [d for d in documents if d.page_content not in existing_texts]
        if not new_docs:
            logger.info("No new documents to upsert (all duplicates)")
        else:
            vectorstore.add_documents(new_docs)
            vectorstore.save_local(faiss_dir)
            logger.info("FAISS index updated at %s (+%d docs)", faiss_dir, len(new_docs))
    else:
        build_faiss_index(documents, faiss_dir)


def load_faiss_index(faiss_dir: str) -> FAISS:
    """Load an existing FAISS index from disk."""
    embeddings = _get_embeddings_model()
    return FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
