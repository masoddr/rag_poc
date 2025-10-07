from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from index_store import load_faiss_index


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Vous êtes un assistant factuel. Répondez en français en vous basant "
    "UNIQUEMENT sur le CONTEXTE fourni. Préférez une réponse courte et "
    "précise, en extrayant les termes exacts du document. Citez les sources "
    "(chemin et page). Si une notion est explicitement nommée (ex: cadre de "
    "référence, ITRF, ECEF), donnez le nom exact. Si l'information n'est pas "
    "dans le contexte, dites que vous ne savez pas."
)


def _format_context(docs: List[Document]) -> str:
    parts: List[str] = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        header = (
            f"Source: {src} | Page: {page}"
            if page is not None
            else f"Source: {src}"
        )
        parts.append(f"[{header}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_llm() -> ChatOllama:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    logger.info("Using Ollama model %s at %s", model, base_url)
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )


def retrieve(
    question: str,
    faiss_dir: str,
    k: int | None = None,
) -> List[Document]:
    top_k = k or int(os.getenv("RETRIEVAL_TOP_K", "4"))
    mmr = os.getenv("RETRIEVAL_MMR", "true").lower() == "true"
    score_threshold = os.getenv("RETRIEVAL_SCORE_THRESHOLD")
    fetch_k_env = os.getenv("RETRIEVAL_FETCH_K")
    try:
        fetch_k = int(fetch_k_env) if fetch_k_env else max(20, top_k * 5)
    except ValueError:
        fetch_k = max(20, top_k * 5)

    vs = load_faiss_index(faiss_dir)

    if score_threshold is not None:
        try:
            threshold_val = float(score_threshold)
        except ValueError:
            threshold_val = None
    else:
        threshold_val = None

    # Optional keyword hints to bias retrieval for critical terms
    hints = os.getenv("RETRIEVAL_HINTS")
    query_text = f"{question} {hints}" if hints else question

    if mmr:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": fetch_k},
        )
    else:
        if threshold_val is not None:
            retriever = vs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": top_k,
                    "score_threshold": threshold_val,
                },
            )
        else:
            retriever = vs.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(query_text)
    logger.info("Retrieved %d documents", len(docs))
    return docs


def _shorten(text: str, max_chars: int = 220) -> str:
    """Return a compact, single-line excerpt limited in length."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def generate_answer(
    question: str,
    context_docs: List[Document],
) -> Tuple[str, List[Dict]]:
    # Truncate context to ~3000 tokens equivalent (rough heuristic by chars)
    context = _format_context(context_docs)
    max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    if len(context) > max_chars:
        context = context[:max_chars]
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"CONTEXTE:\n{context}"
            )
        ),
    ]
    result = llm.invoke(messages)

    sources: List[Dict] = []
    for d in context_docs:
        src_path = d.metadata.get("source")
        page_num = d.metadata.get("page")
        base = os.path.basename(src_path) if src_path else src_path
        sources.append(
            {
                "source": src_path,
                "filename": base,
                "page": page_num,
                "excerpt": _shorten(d.page_content),
            }
        )
    return result.content, sources


def answer_question(question: str, faiss_dir: str) -> str:
    """High-level helper used by CLI/API.

    Retrieves context and queries the LLM to produce an answer string.
    """
    docs = retrieve(question, faiss_dir)
    if not docs:
        return (
            "Je n'ai trouvé aucun passage pertinent dans votre base "
            "documentaire. Veuillez enrichir l'index et réessayer."
        )
    answer, sources = generate_answer(question, docs)
    # Format citations for CLI output (dedupe by filename+page)
    lines: List[str] = [answer, "", "Sources:"]
    seen: set[tuple[str, int | None]] = set()
    for s in sources:
        key = (s.get("filename"), s.get("page"))
        if key in seen:
            continue
        seen.add(key)
        src = s.get("filename") or s.get("source")
        page = s.get("page")
        snippet = s.get("excerpt") or ""
        if page is None:
            lines.append(f"- {src} — \"{snippet}\"")
        else:
            lines.append(f"- {src} | page {page} — \"{snippet}\"")
    return "\n".join(lines)
git