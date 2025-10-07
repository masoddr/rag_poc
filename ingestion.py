from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitParams:
    """Chunking parameters used to split raw documents.

    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
    """

    chunk_size: int = 800
    chunk_overlap: int = 120


def _params_from_env() -> SplitParams:
    """Build split params from environment variables if provided."""
    try:
        size = int(os.getenv("CHUNK_SIZE", "800"))
        overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    except ValueError:
        size, overlap = 800, 120
    return SplitParams(chunk_size=size, chunk_overlap=overlap)


def load_documents(paths: Iterable[str]) -> List[Document]:
    """Load TXT/PDF documents from given paths.

    Args:
        paths: Iterable of filesystem paths to TXT or PDF files.

    Returns:
        List of LangChain Documents with basic metadata.
    """

    documents: List[Document] = []
    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            logger.warning("Path ignored (not a file): %s", path)
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        elif suffix in {".txt", ".md", ".rst"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        else:
            logger.warning("Unsupported file type, skipping: %s", path)
            continue

        # Normalize metadata: keep source path and page if present
        for d in docs:
            meta = dict(d.metadata or {})
            meta.setdefault("source", str(path))
            cleaned = _clean_text(d.page_content)
            # Heuristique: ignorer les pages de sommaire quasi vides
            if _looks_like_toc(cleaned):
                logger.debug(
                    "Skipping likely table-of-contents page: %s", path
                )
                continue
            documents.append(Document(page_content=cleaned, metadata=meta))

    logger.info("Loaded %d raw documents", len(documents))
    return documents


def split_documents(
    documents: List[Document],
    params: SplitParams | None = None,
) -> List[Document]:
    """Split documents into smaller chunks suitable for retrieval.

    Args:
        documents: Raw documents to split.
        params: Optional split parameters; defaults are suitable for
            general text.

    Returns:
        Chunked documents with preserved metadata.
    """

    split_params = params or _params_from_env()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_params.chunk_size,
        chunk_overlap=split_params.chunk_overlap,
        separators=["\n\n", ".", "!", "?", ";", ":"],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _clean_text(text: str) -> str:
    """Light normalization for PDF/TXT content.

    - Remove page-breaks (\f), control chars
    - Collapse multiple spaces/newlines
    """
    # Remove form feed and control chars
    cleaned = text.replace("\f", "\n")
    cleaned = "".join(ch for ch in cleaned if ord(ch) >= 32 or ch in "\n\t")
    # Collapse whitespace
    lines = [" ".join(part.split()) for part in cleaned.splitlines()]
    result = "\n".join(line.strip() for line in lines if line is not None)
    # Guard reasonable bounds
    return result.strip()


def _looks_like_toc(text: str) -> bool:
    """Heuristique simple pour ignorer les pages de sommaire/TOC.

    - Très peu de texte utile, beaucoup de points de conduite ("... 12").
    - Principalement des lignes courtes en majuscules.
    """
    compact = " ".join(text.split())
    if len(compact) < 80:
        return True
    dotted = compact.count(".")
    digits = sum(ch.isdigit() for ch in compact)
    # Densité de points et de chiffres élevée typique d'un sommaire
    return dotted > 30 and digits > 10
