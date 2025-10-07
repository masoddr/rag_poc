from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st
from ingestion import load_documents, split_documents
from index_store import upsert_faiss_index
from rag_pipeline import answer_question


UPLOAD_DIR = Path(".uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="Local RAG (Ollama)", page_icon="🧠", layout="wide"
)
st.title("🧠 Assistant RAG local (Ollama)")

with st.sidebar:
    st.header("Paramètres (simple)")
    default_model = os.getenv("OLLAMA_MODEL", "llama3")
    model = st.text_input(
        "Modèle Ollama",
        value=default_model,
        help="Nom du modèle disponible dans Ollama",
    )
    os.environ["OLLAMA_MODEL"] = model
    faiss_dir = st.text_input(
        "Dossier FAISS", value=os.getenv("FAISS_DIR", ".faiss_index")
    )
    os.environ["FAISS_DIR"] = faiss_dir
    top_k = st.number_input(
        "Top-K",
        min_value=1,
        max_value=10,
        value=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        help="Nombre de passages utilisés pour répondre",
    )
    os.environ["RETRIEVAL_TOP_K"] = str(top_k)
    temperature = st.slider(
        "Température", 0.0, 1.0, float(os.getenv("OLLAMA_TEMPERATURE", "0.2")), 0.05,
        help="Plus bas = plus factuel"
    )
    os.environ["OLLAMA_TEMPERATURE"] = str(temperature)

    with st.expander("Avancé (facultatif)"):
        st.caption("Laissez par défaut si vous débutez")
        use_mmr = st.checkbox("MMR (diversité)", value=False)
        os.environ["RETRIEVAL_MMR"] = "true" if use_mmr else "false"
        score_threshold = st.text_input(
            "Seuil de score (0-1)", value=os.getenv("RETRIEVAL_SCORE_THRESHOLD", "")
        )
        os.environ["RETRIEVAL_SCORE_THRESHOLD"] = score_threshold
        chunk_size = st.number_input(
            "Chunk size", min_value=100, max_value=4000, value=int(os.getenv("CHUNK_SIZE", "800"))
        )
        chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=1000, value=int(os.getenv("CHUNK_OVERLAP", "120"))
        )
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        top_p = st.slider("Top-p", 0.0, 1.0, float(os.getenv("OLLAMA_TOP_P", "0.9")), 0.05)
        os.environ["OLLAMA_TOP_P"] = str(top_p)

st.subheader("1) Upload de documents (TXT / PDF)")
files = st.file_uploader(
    "Glissez vos fichiers ici",
    accept_multiple_files=True,
    type=["txt", "pdf", "md", "rst"],
)
if files:
    saved_paths: List[str] = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        dest.write_bytes(f.read())
        saved_paths.append(str(dest))
    st.success(f"Sauvegardé: {', '.join(saved_paths)}")

st.subheader("2) Construire / Mettre à jour l'index")
if st.button("Construire / Mettre à jour l'index FAISS"):
    if not any(UPLOAD_DIR.iterdir()):
        st.warning(
            "Aucun fichier dans .uploads/. "
            "Uploadez d'abord des documents."
        )
    else:
        docs = load_documents([str(p) for p in UPLOAD_DIR.iterdir()])
        chunks = split_documents(docs)
        upsert_faiss_index(chunks, faiss_dir)
        st.success(f"Index FAISS mis à jour dans: {faiss_dir}")

st.subheader("3) Poser une question")
question = st.text_input("Votre question", value="")
if st.button("Répondre"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        answer = answer_question(question, faiss_dir)
        st.markdown("### Réponse")
        st.write(answer)

st.divider()
with st.expander("Tutoriel (débutant)", expanded=True):
    st.markdown(
        """
        1. Dans la barre de gauche, vérifiez le modèle (ex: `llama3`).
        2. Déposez vos fichiers dans la zone d'upload ci-dessus.
        3. Cliquez sur "Construire / Mettre à jour l'index".
        4. Posez votre question dans la zone de texte.
        5. Ajustez seulement Top-K (plus d'info) et Température (plus factuel) si besoin.
        """
    )
