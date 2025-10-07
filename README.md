# RAG local minimal avec Ollama (Python 3.12)

RAG local simple: FAISS + embeddings HuggingFace + LLM via Ollama. Fourni en CLI, API FastAPI et UI Streamlit.

## Démarrage rapide

1) Installer l’environnement
```bash
conda create -n rag python=3.12 -y && conda activate rag
pip install -r requirements.txt
```

2) Construire l’index à partir de vos documents (TXT/PDF)
```bash
python rag.py build /chemin/vers/doc1.txt /chemin/vers/doc2.pdf
```

3) Poser une question (CLI)
```bash
python rag.py ask "Ma question ?"
```

## Ollama: installation, modèles et réglages

- Installation locale (Linux/Mac):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- Lancer/voir les modèles disponibles:
  ```bash
  ollama list
  ollama pull llama3           # télécharge le modèle si absent
  ollama run llama3            # premier run pour l’initialiser
  ```
- Variables d’environnement utilisées par le projet:
  ```bash
  export OLLAMA_BASE_URL=http://localhost:11434
  export OLLAMA_MODEL=llama3
  ```

### Choisir un modèle (règles simples)

- Ressources limitées (CPU ou <4 Go VRAM): préférez `phi3:mini` ou `llama3.2:3b`.
- GPU modeste (≈4–8 Go VRAM): `llama3:8b` ou `mistral:7b` en quantisation légère.
- GPU confortable (≥12 Go VRAM): `llama3.1:8b` (ou plus grand si dispo).

Astuce: commencez par `llama3` (bon équilibre qualité/coût), puis ajustez selon la latence et la mémoire.

## UI Streamlit (optionnel)

```bash
streamlit run ui_streamlit.py --server.address 0.0.0.0 --server.port 7860
```
- Upload de fichiers TXT/PDF, bouton « Construire / Mettre à jour l’index », puis poser une question.
- Dossier d’uploads par défaut: `.uploads/`.

Variables utiles pour l’UI:
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
export FAISS_DIR=.faiss_index
export RETRIEVAL_TOP_K=4
```

## API FastAPI (local)

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
export EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
export FAISS_DIR=.faiss_index
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Requête de test:
```bash
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "De quoi parle le document ?"}'
```

## Configuration essentielle

- `EMBEDDINGS_MODEL` (défaut: `sentence-transformers/all-MiniLM-L6-v2`)
- `OLLAMA_BASE_URL` (défaut: `http://localhost:11434`)
- `OLLAMA_MODEL` (défaut: `llama3`)
- `FAISS_DIR` (défaut: `.faiss_index`)
- Récupération: `RETRIEVAL_TOP_K` (défaut: `4`), `RETRIEVAL_MMR` (`true/false`), `RETRIEVAL_SCORE_THRESHOLD` (`0.0–1.0`)
- Génération: `OLLAMA_TEMPERATURE` (défaut: `0.2`), `OLLAMA_TOP_P` (défaut: `0.9`)

## Structure du projet (repères)

- `rag.py`: CLI pour `build` (index) et `ask` (question).
- `api.py`: FastAPI `POST /ask`.
- `ui_streamlit.py`: interface simple pour upload/build/ask.
- `ingestion.py`: lecture TXT/PDF et découpage en chunks.
- `index_store.py`: création/chargement FAISS + embeddings.
- `rag_pipeline.py`: retrieval + appel LLM (Ollama).

Notes: la première exécution `ollama run <modele>` télécharge le modèle. Si l’index FAISS est absent, commencez par `rag.py build`.
