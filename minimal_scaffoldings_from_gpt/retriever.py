from typing import Dict, Any, List
from embeddings import embed_texts
from typing import List, Dict, Any
from typing import List
from vector_sore import VectorStore

vs = VectorStore()

# Serialize donor docs to text for embedding


def donor_doc_to_text(doc: Dict[str, Any]) -> str:
    ints = ", ".join(doc.get("interests", []))
    seg = doc.get("segment") or doc.get("donor_quadrant") or ""
    tier = doc.get("prospect_tier") or ""
    lg = doc.get("gifts_3yr", doc.get("lifetime_giving", 0))
    cap = doc.get("capacity_estimate", "?")
    city = doc.get("city", "")
    state = doc.get("state", "")
    aff = doc.get("affinity_score", "?")
    base = (
        f"Prospect: {doc['display_name']} in {city}, {state}. "
        f"Capacity est: {cap}. Affinity/propensity: {aff}. "
        f"Lifetime giving: ${lg}."
    )
    extra = (f" Segment: {seg}. Tier: {tier}." if seg or tier else "")
    interests_txt = (f" Interests: {ints}." if ints else "")
    return base + extra + interests_txt

# Public APIs


def upsert_donor_docs(docs: List[Dict[str, Any]]):
    # Build text + embeddings, and keep rich metadata (segment, tier, scores) for filtering/ranking.
    texts = [donor_doc_to_text(d) for d in docs]
    vecs = embed_texts(texts)

    # Prepare metadata records (copy doc; ensure JSON-serializable types if needed)
    metas = []
    for d in docs:
        m = dict(d)
        # ensure required fields exist
        m.setdefault("type", "donor_profile")
        m.setdefault("provenance", {})
        metas.append(m)

    vs.upsert([
        {
            "id": str(d["donor_id"]),
            "text": t,
            "metadata": m,
            "vector": v,
        } for d, t, v, m in zip(docs, texts, vecs, metas)
    ])


def search_candidates(query: str, filters: Dict[str, Any], k: int): (query: str, filters: Dict[str, Any], k: int):
    where = {}
    if city := filters.get("city"):
        where["city"] = {"$eq": city}
    if state := filters.get("state"):
        where["state"] = {"$eq": state}
    if buckets := filters.get("capacity_bucket"):
        where["capacity_estimate"] = {"$in": buckets}
    # NOTE: interests filter could be done via keyword match or custom where if stored normalized
    res = vs.hybrid_search(query=query, where=where, k=k)
    return res
