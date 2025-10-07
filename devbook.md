Développe une architecture complète de RAG documentaire en Python, adaptée à un LLM local **Ollama** (modèle au choix), afin d’interroger des documents personnels via CLI, API et UI Streamlit.

1. Objectifs et périmètre
   - Python 3.12+
   - Orchestration RAG avec `langchain` (alternativement `llama-index` possible plus tard)
   - Embeddings via `sentence-transformers` (ex: `sentence-transformers/all-MiniLM-L6-v2`)
   - Vector store local: `FAISS` (persistant sur disque)
   - LLM servi par Ollama via `OLLAMA_BASE_URL` et `OLLAMA_MODEL`
   - Interfaces: CLI (`rag.py`), API FastAPI (`POST /ask`), UI Streamlit (upload, build index, ask)

2. Architecture logique (Domain / Adapter / Interface)
   - Domain
     - Découpage textuel (chunks) et schéma de métadonnées minimal (source, page)
     - Pipeline RAG: retrieve top-k, compose prompt, interroger le LLM Ollama
   - Adapters
     - Ingestion: loaders TXT/PDF (TextLoader, PyPDFLoader)
     - Embeddings HuggingFace + FAISS pour l’indexation et la recherche
   - Interfaces
     - CLI `rag.py`: build/ask
     - FastAPI `api.py`: `POST /ask` avec JSON {"question": str}
     - Streamlit `ui_streamlit.py`: upload de fichiers, construction/MAJ d’index, question/réponse

3. Spécifications techniques
   - Ingestion
     - Formats supportés: TXT, PDF (extensible)
     - Découpage: `RecursiveCharacterTextSplitter` (par défaut: chunk_size=800, chunk_overlap=120)
     - Normalisation d’encodage (UTF-8), extraction texte PDF via `pypdf`
   - Indexation
     - Embeddings: `HuggingFaceEmbeddings` (modèle configurable via env)
     - FAISS persistant dans un dossier `FAISS_DIR` (par défaut: `.faiss_index`)
     - Opérations: création, chargement, upsert (ajout incrémental)
   - Récupération + Génération
     - Retriever FAISS top_k configurable (par défaut: 4)
     - LLM: `ChatOllama` (ou `Ollama`) via `OLLAMA_BASE_URL` et `OLLAMA_MODEL`
     - Prompt de réponse citant les sources; mode strict: ne pas halluciner si pas de contexte pertinent
   - Observabilité
     - Logging avec `logging` (niveau via `LOG_LEVEL`)

4. Variables d’environnement (fichier `.env`)
   - `EMBEDDINGS_MODEL` (def: `sentence-transformers/all-MiniLM-L6-v2`)
   - `OLLAMA_BASE_URL` (def: `http://localhost:11434`)
   - `OLLAMA_MODEL` (def: `llama3`)
   - `FAISS_DIR` (def: `.faiss_index`)
   - `RETRIEVAL_TOP_K` (def: `4`)
   - `LOG_LEVEL` (def: `INFO`)

5. Modules
   - `ingestion.py`: chargement TXT/PDF, splitting, validation basique
   - `index_store.py`: gestion FAISS + embeddings, persistance disque
   - `rag_pipeline.py`: retrieval, formatting prompt, appel LLM Ollama
   - `api.py`: FastAPI `POST /ask`
   - `ui_streamlit.py`: UI pour upload, build/upsert index, et Q/A
   - `rag.py`: CLI existante (build/ask)

6. API locale
   - Endpoint: `POST /ask`
     - Body: `{ "question": "..." }`
     - Réponse: `{ "answer": str, "sources": [{"source": str, "page": int|null}] }`
   - Codes d’erreur: 400 si index introuvable; 422 si payload invalide
   - Exemple curl:
     ```bash
     curl -s -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "De quoi parle le document ?"}'
     ```

7. UI Streamlit
   - Pages
     - Upload fichiers (TXT/PDF) → stockés dans `.uploads/`
     - Bouton "Construire / Mettre à jour l’index" → upsert dans FAISS
     - Zone de question → affiche réponse + sources
   - Paramètres
     - Sélection du modèle Ollama et du top_k

8. Dockerisation (bonus)
   - `docker-compose.yml` avec services:
     - `ollama` (expose 11434)
     - `api` (FastAPI) dépend de `ollama`
   - Variables d’environnement montées depuis `.env`

9. Qualité et bonnes pratiques
   - PEP8/PEP484, typage explicite, docstrings courts et clairs
   - Séparation Domain/Adapter/Interface
   - Tests unitaires: loaders, splitter, fonctions FAISS, formatage du prompt
   - Gestion d’erreurs contrôlée; pas d’effets de bord non maîtrisés

10. Commandes clés
   - CLI: `python rag.py build <paths...>` puis `python rag.py ask "<question>"`
   - API: `uvicorn api:app --reload`
   - UI: `streamlit run ui_streamlit.py`
