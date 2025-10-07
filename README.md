# Minimal Local RAG with Ollama (Python 3.12)

This repo provides a simple, local RAG setup using FAISS + HuggingFace embeddings and an Ollama-backed LLM, with both a CLI and a FastAPI server.

## Mode d'emploi pour collègues (3 étapes)

1) Construire l'index à partir de vos documents (TXT/PDF)
```bash
python rag.py build /chemin/vers/doc1.txt /chemin/vers/doc2.pdf
```
2) Poser une question en CLI
```bash
python rag.py ask "Ma question ?"
```

## Quickstart

### 1) Create env and install deps

```bash
# optional: conda env as per request
conda create -n rag python=3.12 -y && conda activate rag
pip install -r requirements.txt
# Optionnel: cp .env.example .env (si présent) ou définir les variables ci-dessous
```

### 2) Build the index

```bash
# Place your documents anywhere (TXT or PDF)
python rag.py build path/to/doc1.txt path/to/doc2.pdf
```

### 3) Ask a question via CLI

```bash
python rag.py ask "Quelle est la mission du projet ?"
```


## Interface Streamlit (optionnelle)

```bash
pip install -r requirements.txt  # si pas déjà fait
streamlit run ui_streamlit.py --server.address 0.0.0.0 --server.port 7860
```

- Onglet: uploadez des fichiers TXT/PDF, cliquez sur "Construire / Mettre à jour l'index", puis posez une question.
- Les fichiers uploadés sont sauvegardés dans `.uploads/` par défaut.

Variables utiles pour Streamlit:
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
export FAISS_DIR=.faiss_index
export RETRIEVAL_TOP_K=4
```

### Mini-tutoriel (débutant)

1) Ouvrez l'UI: `streamlit run ui_streamlit.py`.
2) Dans la barre gauche, laissez les valeurs par défaut (Modèle, Top-K=4, Température=0.2).
3) Uploadez vos fichiers (TXT/PDF) puis cliquez sur "Construire / Mettre à jour l'index".
4) Posez votre question dans la zone prévue.
5) Si la réponse manque de détails, augmentez Top-K à 5 ou 6. Si elle dérive, baissez Température.


## Structure

- `ingestion.py`: lit les fichiers TXT/PDF et les découpe en petits morceaux (chunks) utilisables pour la recherche.
- `index_store.py`: crée l'index FAISS à partir des chunks et le recharge depuis le disque; configure le modèle d'embeddings.
- `rag_pipeline.py`: assemble la recherche (retriever) et le modèle Ollama pour générer une réponse basée sur les passages trouvés.
  - Paramètres avancés via env: `RETRIEVAL_TOP_K`, `RETRIEVAL_MMR` (true/false),
    `RETRIEVAL_SCORE_THRESHOLD` (0-1), `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`.
- `api.py`: petit serveur FastAPI qui expose `POST /ask` pour poser une question via HTTP.
- `rag.py`: outil en ligne de commande pour construire l'index (`build`) et poser une question (`ask`).
- `.env.example`: variables d'environnement par défaut (modèle d'embed, modèle Ollama, dossier d'index).

## Config

Environment variables (can be set in `.env`):
- `EMBEDDINGS_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3`)
- `FAISS_DIR` (default: `.faiss_index`)
 - `RETRIEVAL_TOP_K` (default: `4`)
 - `RETRIEVAL_MMR` (optional; `true`/`false`)
 - `RETRIEVAL_SCORE_THRESHOLD` (optional; `0.0-1.0`)
 - `CHUNK_SIZE` (default: `800`), `CHUNK_OVERLAP` (default: `120`)
 - `OLLAMA_TEMPERATURE` (default: `0.2`), `OLLAMA_TOP_P` (default: `0.9`)

Exemple d'exécution API locale:
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
export EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
export FAISS_DIR=.faiss_index
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Puis, requête HTTP:
```bash
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "De quoi parle le document ?"}'
```

## Notes
- Ensure the Ollama model is pulled: `ollama run llama3` once, or rely on container to pull.
- If index is missing, the API returns 400 with a clear message. Build first via CLI.
- PDFs are parsed with `pypdf` via `PyPDFLoader`.
