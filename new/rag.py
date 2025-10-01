import os
import textwrap
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient

CHROMA_PATH = r"C:\Rakshya\Trip Advisor\wealth_engine_chroma"  # ABSOLUTE path
COLLECTION = "wealth_engine_prospects_scores"

client = PersistentClient(path=CHROMA_PATH)
# no embedding_fn needed for querying
col = client.get_or_create_collection(name=COLLECTION)


def retrieve(
    question: str,
    k: int = 5,
    city: Optional[str] = None,
    weps_range: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Semantic + optional metadata filtering."""
    where = None
    clauses = []
    if city:
        clauses.append({"city": city})
    if weps_range:
        lo, hi = weps_range
        # NOTE: Your ingest used "WEPS" uppercase; use the exact key you stored.
        clauses.append({"WEPS": {"$gte": float(lo)}})
        clauses.append({"WEPS": {"$lte": float(hi)}})
    if clauses:
        where = {"$and": clauses} if len(clauses) > 1 else clauses[0]

    res = col.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where
    )
    # print(question, k, where)  # DEBUG
    # print(res)
    return res


def build_prompt(question: str, hits: Dict[str, Any]) -> str:
    """Ground the LLM with retrieved context; forbid unsupported claims."""
    blocks = []
    docs = hits.get("documents", [[]])[0] if hits.get("documents") else []
    metas = hits.get("metadatas", [[]])[0] if hits.get("metadatas") else []
    dists = hits.get("distances", [[]])[0] if hits.get("distances") else []

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        blocks.append(f"[{i}] distance={dist:.4f}\nDOC: {doc}\nMETA: {meta}\n")

    context = "\n\n".join(blocks) if blocks else "NO_MATCHES"
    sys = (
        "You are a cautious assistant. Answer ONLY from the provided context.\n"
        "If the context is insufficient, say 'I don't know from the current records.'\n"
        "Do not invent names, scores, or locations.\n"
        "Format your answer in titles\n"
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer succinctly:"
    return sys + "\n---\n" + user
