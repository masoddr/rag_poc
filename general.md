## 1. Build index
# 🧠 Règles RAG – Construction d’Index (PDF/TXT)

Lorsque tu construis ou modifies la partie "build index" du projet RAG (ingestion, embeddings, FAISS):

1. **Ingestion**
   - Utilise `PyPDFLoader` pour les PDF et `TextLoader` pour les TXT.
   - Nettoie le texte : retire sauts de page, caractères de contrôle et espaces multiples.
   - Conserve les métadonnées essentielles : `source`, `page`.

2. **Chunking**
   - Utilise `RecursiveCharacterTextSplitter` avec :
     ```python
     chunk_size=800
     chunk_overlap=120
     separators=["\n\n", ".", "!", "?", ";", ":"]
     ```
   - Découpe uniquement après phrase complète ou paragraphe.
   - Évite les chunks trop longs (>1000) ou trop courts (<200).

3. **Embeddings**
   - Modèle par défaut : `sentence-transformers/all-MiniLM-L6-v2`.
   - Si corpus technique, proposer `intfloat/e5-large-v2` ou `multi-qa-mpnet-base-dot-v1`.

4. **Indexation**
   - Stocke l’index FAISS persistant (ex: `.faiss_index/`).
   - Toujours sauvegarder après `build` ou `upsert`.
   - Si index existe : charge, puis ajoute (`upsert_documents`) au lieu de recréer.
   - Vérifie qu’aucun doublon n’est inséré (basé sur hash du contenu).

5. **Logs**
   - Log le nombre de documents, chunks, et la taille finale de l’index.
   - En cas d’erreur d’encodage PDF, ignorer le fichier et continuer.

6. **Qualité**
   - Vérifie que `load_faiss_index()` ne casse pas si dossier vide.
   - Le code doit être typé (PEP484) et suivre PEP8.
   - Le pipeline doit être reproductible et idempotent.

7. **Test rapide**
   - Après build, vérifier :
     ```python
     db.similarity_search("mot clé test", k=3)
     ```
     pour s’assurer que l’index renvoie bien les bons passages.

# Objectif
Construire un index FAISS robuste, propre et réutilisable, garantissant un retrieval précis pour les futures requêtes de Q/A.

## 2. Interrogation de l'index
# 🗣️ RAG – Interrogation de l’Index (Retrieval + Prompt)

Lorsque tu modifies ou construis la partie "question answering" du pipeline RAG :

1. **Chargement**
   - Charge l’index FAISS via `load_faiss_index()`.
   - Lève une erreur claire si le dossier `.faiss_index` est manquant.
   - Instancie le retriever :
     ```python
     retriever = db.as_retriever(search_kwargs={"k": int(os.getenv("RETRIEVAL_TOP_K", 4))})
     ```

2. **Recherche**
   - Récupère les top-k documents les plus pertinents.
   - Vérifie que `k` reste modéré (3–6).
   - Utilise la similarité cosinus par défaut (celle de FAISS).
   - Si aucun résultat pertinent, renvoie une réponse neutre (“Je ne sais pas”).

3. **Assemblage du contexte**
   - Concatène les passages trouvés avec leurs métadonnées :
     ```python
     context = "\n\n".join([f"Source: {d.metadata.get('source')} (page {d.metadata.get('page', '?')})\n{d.page_content}" for d in docs])
     ```
   - Tronque le contexte si > 3 000 tokens avant envoi au LLM.

4. **Prompt**
   - Formate toujours le prompt ainsi :
     ```python
     f"""
     Tu es un assistant factuel. Réponds uniquement à partir du contexte ci-dessous.
     Si le contexte ne contient pas la réponse, dis simplement "Je ne sais pas".

     Contexte:
     {context}

     Question:
     {question}

     Réponds de façon claire et concise, en citant les sources utilisées.
     """
     ```
   - Évite toute reformulation ou invention hors contexte.

5. **LLM (Ollama)**
   - Utilise `ChatOllama` ou `Ollama` selon la version LangChain.
   - Paramètres via `.env` :
     ```bash
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_MODEL=llama3.1:8b
     ```
   - Si la VRAM est limitée, recommande `mistral:7b` ou `phi3:mini`.

6. **Post-traitement**
   - Retourne un JSON structuré :
     ```python
     {
       "answer": str,
       "sources": [{"source": str, "page": int|null}]
     }
     ```
   - Nettoie les réponses : pas de markdown parasite ni de doublons dans les sources.

7. **Observabilité**
   - Log la question, le nombre de chunks récupérés et la durée totale.
   - Mesure la latence d’appel LLM pour évaluer les perfs.

# Objectif
Obtenir des réponses précises, sourcées et sans hallucination, en combinant retrieval FAISS + prompting contrôlé vers le LLM Ollama.
