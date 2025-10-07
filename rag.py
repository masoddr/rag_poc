import argparse
import os
from typing import List

from ingestion import load_documents, split_documents
from index_store import build_faiss_index, load_faiss_index, upsert_faiss_index
from rag_pipeline import answer_question


def cmd_build(paths: List[str], faiss_dir: str) -> None:
    docs = load_documents(paths)
    chunks = split_documents(docs)
    upsert_faiss_index(chunks, faiss_dir)
    print(f"Built FAISS index at: {faiss_dir}")


def cmd_ask(question: str, faiss_dir: str) -> None:
    answer = answer_question(question, faiss_dir)
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal local RAG helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build FAISS index from documents")
    p_build.add_argument("paths", nargs="+", help="Paths to documents (TXT/PDF)")
    p_build.add_argument("--faiss-dir", default=os.getenv("FAISS_DIR", ".faiss_index"))

    p_ask = sub.add_parser("ask", help="Ask a question against the index")
    p_ask.add_argument("question")
    p_ask.add_argument("--faiss-dir", default=os.getenv("FAISS_DIR", ".faiss_index"))

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args.paths, args.faiss_dir)
    elif args.command == "ask":
        cmd_ask(args.question, args.faiss_dir)


if __name__ == "__main__":
    main()


