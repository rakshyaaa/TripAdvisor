# Project layout
#
# rag_trip_advisor/
# ├─ app.py                 # FastAPI app exposing chat + tools
# ├─ config.py              # Settings (env-driven)
# ├─ embeddings.py          # Embed text rows
# ├─ ingest.py              # Export -> transform -> embed -> upsert
# ├─ retriever.py           # Hybrid retrieval (metadata + vectors)
# ├─ scorer.py              # Priority scoring + itinerary packing
# ├─ tools.py               # Tool functions the agent can call
# ├─ prompts.py             # System & tool-call prompts
# ├─ schemas.py             # Pydantic models
# ├─ storage/
# │   ├─ vector_store.py    # Abstraction over pgvector / Chroma
# │   └─ local/             # Default local sqlite/chroma artifacts
# ├─ sql/
# │   └─ views.sql          # Safe read-only views (example)
# ├─ requirements.txt
# └─ README.md

# -------------------------------
# requirements.txt
# -------------------------------
# FastAPI and uvicorn for the API
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from typing import Iterable
import json
import hashlib
import argparse
import re
from retriever import upsert_donor_docs
import pandas as pd
from tools import tool_build_itinerary, tool_search_candidates
from prompts import SYSTEM_PROMPT
from fastapi import FastAPI
from scorer import priority_score, pack_itinerary
from retriever import search_candidates
from typing import Dict, Any, List
import math
from datetime import datetime
from embeddings import embed_texts
from storage.vector_store import VectorStore
from dataclasses import dataclass
from typing import List, Dict, Any
from config import settings
from typing import List
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
from pydantic import BaseModel
fastapi
uvicorn
# Data
pandas
pydantic
# Vector DB (choose one; default to Chroma local)
chromadb
# If using Postgres+pgvector instead, uncomment below and configure storage/vector_store.py
# psycopg2-binary
# sqlalchemy
# External embeddings provider (OpenAI shown as an example)
openai >= 1.40.0
python-dotenv
# SQL Server (for direct ingestion from your view)
sqlalchemy >= 2.0
pyodbc

# -------------------------------
# config.py
# -------------------------------


class Settings(BaseModel):
    # RAG knobs
    top_k: int = int(os.getenv("TOP_K", 30))
    rerank_k: int = int(os.getenv("RERANK_K", 12))
    slots_per_day: int = int(os.getenv("SLOTS_PER_DAY", 6))
    recency_half_life_days: int = int(os.getenv("RECENCY_HALF_LIFE_DAYS", 270))

    # Storage
    storage_backend: str = os.getenv(
        "STORAGE_BACKEND", "chroma")  # or "pgvector"
    storage_path: str = os.getenv("STORAGE_PATH", "./storage/local")

    # Embeddings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-3-small")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # SQL Server ODBC connection string (optional, for direct ingestion)
    # Example (Windows):
    #   Driver={ODBC Driver 18 for SQL Server};Server=tcp:myserver.database.windows.net,1433;Database=CRM_Advance;UID=...;PWD=...;Encrypt=yes;TrustServerCertificate=no
    # Example (Linux):
    #   Driver={ODBC Driver 18 for SQL Server};Server=tcp:myserver,1433;Database=CRM_Advance;UID=...;PWD=...;Encrypt=yes;TrustServerCertificate=yes
    mssql_odbc_connect: str | None = os.getenv("MSSQL_ODBC_CONNECT")

    # API
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))


settings = Settings()

# -------------------------------
# schemas.py
# -------------------------------


class DonorDoc(BaseModel):
    type: str = "donor_profile"
    donor_id: int
    display_name: str
    city: str
    state: str
    capacity_estimate: int
    affinity_score: float
    gifts_3yr: float
    last_gift_dt: Optional[str] = None
    last_touch_dt: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    donor_id: int
    display_name: str
    city: str
    state: str
    capacity_estimate: int
    affinity_score: float
    gifts_3yr: float
    last_touch_dt: Optional[str]
    score: float
    provenance: Dict[str, Any]


class SearchFilters(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_bucket: Optional[List[int]] = None
    interests: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = 30


class ItineraryMeeting(BaseModel):
    time: str
    donor_id: int
    display_name: str
    rationale: str
    score_breakdown: Dict[str, Any]
    provenance: Dict[str, Any]


class DayPlan(BaseModel):
    date: str
    city: str
    meetings: List[ItineraryMeeting]


class ItineraryResponse(BaseModel):
    itinerary: List[DayPlan]
    coverage_notes: str | None = None
    data_gaps: List[str] = Field(default_factory=list)


# -------------------------------
# embeddings.py
# -------------------------------

# Minimal provider wrapper (OpenAI by default)


def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(
            model=settings.embedding_model, input=texts)
        return [d.embedding for d in resp.data]
    else:
        raise NotImplementedError(
            f"Embedding provider '{provider}' not supported in minimal scaffold.")


# -------------------------------
# storage/vector_store.py
# -------------------------------


@dataclass
class VectorDoc:
    id: str
    text: str
    metadata: Dict[str, Any]
    vector: List[float] | None = None


class VectorStore:
    def __init__(self):
        backend = settings.storage_backend
        if backend == "chroma":
            import chromadb
            self.client = chromadb.PersistentClient(path=settings.storage_path)
            self.coll = self.client.get_or_create_collection("donor_docs")
        else:
            raise NotImplementedError(
                "Only 'chroma' backend implemented in minimal scaffold.")

    def upsert(self, docs: List[VectorDoc]):
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        metas = [d.metadata for d in docs]
        vecs = [d.vector for d in docs] if docs and docs[0].vector is not None else None
        self.coll.upsert(ids=ids, documents=texts,
                         metadatas=metas, embeddings=vecs)

    def hybrid_search(self, query: str, where: Dict[str, Any] | None, k: int):
        # Chroma supports where filtering + embedding search; we'll use embedding w/ where.
        return self.coll.query(query_texts=[query], n_results=k, where=where or {})


# -------------------------------
# retriever.py
# -------------------------------

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


# -------------------------------
# scorer.py
# -------------------------------


def days_since(date_str: str | None) -> float:
    if not date_str:
        return 9999
    return (datetime.utcnow() - datetime.fromisoformat(date_str)).days


def recency_decay(days: float, half_life: int) -> float:
    return math.exp(-math.log(2) * days / max(1, half_life))


def priority_score(capacity: int, affinity: float, gifts_3yr: float, last_touch: str | None, half_life_days: int = 270,
                   weights: dict | None = None) -> tuple[float, dict]:
    w = {"capacity": 0.30, "affinity": 0.35, "recency": 0.20, "gifts": 0.15}
    if weights:
        w.update(weights)
    d = days_since(last_touch)
    features = {
        "capacity": capacity / 5.0,  # normalize 1-5 → 0-1
        "affinity": float(affinity),
        "recency": recency_decay(d, half_life_days),
        # cap at $100k for normalization
        "gifts": min(gifts_3yr / 100000.0, 1.0),
        "recency_days": d,
    }
    score = sum(features[k] * w[k] for k in ["capacity", "affinity", "recency", "gifts"]) \
            + 0.0  # placeholder for travel_penalty subtraction during itinerary packing
    return score, features


def pack_itinerary(cands: list[dict], city: str, dates: list[str], slots_per_day: int = 6):
    # Greedy bin pack by date; assume all candidates are in-city for a minimal scaffold.
    meetings_per_day = {d: [] for d in dates}
    pointer = 0
    for d in dates:
        for _ in range(slots_per_day):
            if pointer >= len(cands):
                break
            meetings_per_day[d].append(cands[pointer])
            pointer += 1
    day_plans = []
    for d in dates:
        ms = [
            {
                "time": f"{9 + i:02d}:00",
                "donor_id": c["donor_id"],
                "display_name": c["display_name"],
                "rationale": c["rationale"],
                "score_breakdown": c["score_breakdown"],
                "provenance": c["provenance"],
            }
            for i, c in enumerate(meetings_per_day[d])
        ]
        day_plans.append({"date": d, "city": city, "meetings": ms})
    return day_plans


# -------------------------------
# tools.py
# -------------------------------


def tool_search_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = payload.get("query", "")
    filters = payload.get("filters", {})
    k = int(payload.get("k", settings.top_k))
    res = search_candidates(query, filters, k=k)
    # Normalize chroma result → list of metadata dicts + raw scores
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        out.append(meta)
    return out


def tool_build_itinerary(payload: Dict[str, Any]) -> Dict[str, Any]:
    city = payload.get("city")
    dates = payload.get("dates", [])
    query = payload.get("query", "")
    filters = payload.get("filters", {})
    k = int(payload.get("k", settings.rerank_k))

    raw = tool_search_candidates(
        {"query": query, "filters": filters, "k": settings.top_k})

    # Score & rank
    scored = []
    for r in raw:
        score, feats = priority_score(
            capacity=r.get("capacity_estimate", 1),
            affinity=r.get("affinity_score", 0.0),
            gifts_3yr=r.get("gifts_3yr", 0.0),
            last_touch=r.get("last_touch_dt"),
            half_life_days=settings.recency_half_life_days,
        )
        scored.append({
            "donor_id": r["donor_id"],
            "display_name": r["display_name"],
            "city": r["city"],
            "state": r["state"],
            "score": score,
            "rationale": (
                f"Affinity {r.get('affinity_score', 0):.2f}, capacity {r.get('capacity_estimate', 1)}, "
                f"3yr gifts ${r.get('gifts_3yr', 0):,.0f}, last touch {r.get('last_touch_dt', 'unknown')}"
            ),
            "score_breakdown": feats,
            "provenance": r.get("provenance", {}),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:k]
    days = pack_itinerary(top, city=city, dates=dates,
                          slots_per_day=settings.slots_per_day)

    return {"itinerary": days, "coverage_notes": ("Auto-packed; adjust slots or filters as needed.")}


# -------------------------------
# prompts.py
# -------------------------------
SYSTEM_PROMPT = (
    "You are a fundraising trip advisor. Propose donor visit plans that maximize expected gift outcomes "
    "while respecting travel and scheduling constraints. Always show rationale and provenance. "
    "Never fabricate facts; if data is missing, say so."
)

# -------------------------------
# app.py
# -------------------------------

app = FastAPI(title="RAG Trip Advisor", version="0.1")


class ChatRequest(BaseModel):
    city: str
    dates: List[str]
    query: str
    filters: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/build-itinerary")
def build_itinerary(req: ChatRequest):
    payload = {
        "city": req.city,
        "dates": req.dates,
        "query": req.query,
        "filters": req.filters,
    }
    res = tool_build_itinerary(payload)
    return res


@app.post("/search")
def search(req: Dict[str, Any]):
    return tool_search_candidates(req)


# -------------------------------
# ingest.py (example: from CSV → donor docs)
# -------------------------------

# Expect a CSV with columns: donor_id,display_name,city,state,capacity_estimate,affinity_score,gifts_3yr,last_touch_dt,interests
# Use your own export process to generate this safely from your DB view(s).


def ingest_from_csv(path: str):
    df = pd.read_csv(path)
    docs = []
    for _, r in df.iterrows():
        interests = [s.strip() for s in str(
            r.get("interests", "")).split("|") if s and s.strip()]
        docs.append({
            "type": "donor_profile",
            "donor_id": int(r["donor_id"]),
            "display_name": r["display_name"],
            "city": r["city"],
            "state": r["state"],
            "capacity_estimate": int(r["capacity_estimate"]),
            "affinity_score": float(r["affinity_score"]),
            "gifts_3yr": float(r.get("gifts_3yr", 0.0)),
            "last_touch_dt": r.get("last_touch_dt"),
            "interests": interests,
            "provenance": {"source": "csv_export", "file": path},
        })
    upsert_donor_docs(docs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ingest.py ./exports/donor_profiles.csv")
        raise SystemExit(1)
    ingest_from_csv(sys.argv[1])

# -------------------------------
# ingest_sqlserver.py (ingest from SQL Server view)
# -------------------------------


VIEW_DEFAULT = "dbo.view_wealth_engine_prospect_scores"

# split interests on common delimiters
DELIMS = re.compile(r"[;,|/]+")

# ---- Helpers to map SQL → profile doc ---------------------------------------


def _capacity_estimate_from_band(band: str | None) -> int:
    """Map capacity band label (A/B/C) to a 1–5 estimate for scoring.
    A → 5, B → 3, C/other → 1
    """
    if not band:
        return 3
    b = str(band).strip().upper()
    if b.startswith("A-"):
        return 5
    if b.startswith("B-"):
        return 3
    return 1


def _interests_from_inclination(value: str | None, board_membership: str | None) -> list[str]:
    out: list[str] = []
    if value:
        parts = [p.strip()
                         for p in DELIMS.split(str(value)) if p and p.strip()]
        out.extend(parts)
    if board_membership:
        bm = str(board_membership).strip()
        if bm and bm.lower() not in {"0", "no", "false", "none"}:
            out.append(f"Board: {bm}")
    # de-duplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


ROW_HASH_FIELDS = [
    "display_name", "city", "state", "capacity_estimate", "affinity_score",
    "wealth_score", "propensity_score", "influence_score", "givingCapacity_score",
    "lifetime_giving", "lifetime_giving_label", "donor_quadrant", "capacity_band",
    "segment", "overall_decile", "segment_decile", "prospect_tier", "interests",
]


def _doc_hash(doc: dict) -> str:
    """Stable hash of the subset of fields that affect retrieval/answers."""
    payload = {k: doc.get(k) for k in ROW_HASH_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _row_to_doc(r: pd.Series) -> dict:
    # Map SQL view → donor doc used by the RAG store
    capacity_estimate = _capacity_estimate_from_band(r.get("capacity_band"))
    affinity_score = float(r.get("propensity_score") or 0.0) / 100.0
    lifetime_giving = float(r.get("lifetime_giving") or 0.0)
    interests = _interests_from_inclination(
        r.get("inclinationAffiliation"), r.get("boardMembership"))

    doc = {
        "type": "donor_profile",
        "donor_id": int(r.get("primary_id")),
        "display_name": str(r.get("person_name") or "Unknown"),
        "city": str(r.get("city") or ""),
        "state": str(r.get("state") or ""),
        "capacity_estimate": int(capacity_estimate),
        "affinity_score": float(affinity_score),
        # Use lifetime giving as a proxy for gifts_3yr if you don't have a 3-year window here.
        "gifts_3yr": float(lifetime_giving),
        "last_touch_dt": None,  # not present in this view; can be joined later
        "interests": interests,
        # Carry rich metadata for filtering/explanations
        "wealth_score": float(r.get("wealth_score") or 0.0),
        "propensity_score": float(r.get("propensity_score") or 0.0),
        "influence_score": float(r.get("influence_score") or 0.0),
        "givingCapacity_score": float(r.get("givingCapacity_score") or 0.0),
        "lifetime_giving": lifetime_giving,
        "lifetime_giving_label": str(r.get("lifetime_giving_label") or ""),
        "donor_quadrant": str(r.get("donor_quadrant") or ""),
        "capacity_band": str(r.get("capacity_band") or ""),
        "segment": str(r.get("segment") or ""),
        "overall_decile": int(r.get("overall_decile") or 0),
        "segment_decile": int(r.get("segment_decile") or 0),
        "prospect_tier": str(r.get("prospect_tier") or ""),
        "primary_prospect_manager_list": str(r.get("primary_prospect_manager_list") or ""),
        "provenance": {"view": VIEW_DEFAULT, "pk": {"primary_id": int(r.get("primary_id"))}},
    }
    doc["row_hash"] = _doc_hash(doc)
    return doc


# ---- Fetch from SQL Server ---------------------------------------------------

def _fetch_df(view: str, where: str | None, limit: int | None) -> pd.DataFrame:
    if not settings.mssql_odbc_connect:
        raise RuntimeError(
            "MSSQL_ODBC_CONNECT env var not set. See README for an example.")

    conn = "mssql+pyodbc:///?odbc_connect=" + \
        quote_plus(settings.mssql_odbc_connect)
    engine = create_engine(conn)

    sql = f"SELECT * FROM {view}"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" ORDER BY primary_id OFFSET 0 ROWS FETCH NEXT {int(limit)} ROWS ONLY"

    with engine.begin() as cxn:
        df = pd.read_sql(text(sql), cxn)
    # drop null ids
    df = df[pd.notnull(df["primary_id"])].copy()
    return df


# ---- Vector sync utilities ---------------------------------------------------

def _get_existing_hashes(vs: VectorStore, ids: Iterable[str]) -> dict[str, str | None]:
    """Return {id: row_hash or None} for those ids present in the vector store."""
    id_list = list(ids)
    if not id_list:
        return {}
    try:
        # type: ignore[attr-defined]
        data = vs.coll.get(ids=id_list, include=["metadatas"])
    except Exception:
        # If backend doesn't support .coll.get, assume empty
        return {}
    existing: dict[str, str | None] = {}
    for i, md in zip(data.get("ids", []), data.get("metadatas", [])):
        if md is None:
            existing[str(i)] = None
        else:
            existing[str(i)] = md.get(
                "row_hash") if isinstance(md, dict) else None
    return existing


def _delete_missing(vs: VectorStore, current_sql_ids: set[str]) -> int:
    """Delete vector docs whose ids are NOT in SQL anymore. Returns count deleted."""
    try:
        all_data = vs.coll.get(include=["ids"])  # type: ignore[attr-defined]
        existing_ids = set(map(str, all_data.get("ids", [])))
    except Exception:
        # Backend might not expose list; skip
        return 0
    to_delete = list(existing_ids - current_sql_ids)
    if not to_delete:
        return 0
    vs.coll.delete(ids=to_delete)  # type: ignore[attr-defined]
    return len(to_delete)


# ---- Public entrypoint -------------------------------------------------------

def ingest_from_sqlserver(
    view: str = VIEW_DEFAULT,
    where: str | None = None,
    limit: int | None = None,
    batch_size: int = 1000,
    skip_unchanged: bool = True,
    sync_delete: bool = False,
):
    df = _fetch_df(view=view, where=where, limit=limit)

    # Map rows → docs
    docs = [_row_to_doc(r) for _, r in df.iterrows()]

    # Optionally skip unchanged by comparing row_hash to what's in vector store
    vs = VectorStore()
    if skip_unchanged and docs:
        existing_hashes = _get_existing_hashes(
            vs, [str(d["donor_id"]) for d in docs])
        filtered_docs = []
        skipped = 0
        for d in docs:
            cur = existing_hashes.get(str(d["donor_id"]))
            if cur and cur == d["row_hash"]:
                skipped += 1
                continue
            filtered_docs.append(d)
        docs = filtered_docs
    else:
        skipped = 0

    # Upsert in batches to control memory/latency
    total = 0
    if docs:
        for i in range(0, len(docs), batch_size):
            chunk = docs[i: i + batch_size]
            upsert_donor_docs(chunk)
            total += len(chunk)

    # Optionally remove stale docs
    deleted = 0
    if sync_delete:
        sql_ids = set(map(lambda x: str(int(x)), df["primary_id"].tolist()))
        deleted = _delete_missing(vs, sql_ids)

    return {"rows_fetched": int(df.shape[0]), "upserts": total, "skipped": skipped, "deleted": deleted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest prospects from SQL Server view into the vector store")
    parser.add_argument("--view", default=VIEW_DEFAULT,
                        help="Qualified view name")
    parser.add_argument("--where", default=None,
                        help="Optional WHERE clause, e.g. state = 'TX' AND city = 'Austin'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional row limit (uses OFFSET/FETCH)")
    parser.add_argument("--batch-size", type=int,
                        default=1000, help="Embed/upsert batch size")
    parser.add_argument("--no-skip-unchanged", action="store_true",
                        help="Re-embed everything even if unchanged")
    parser.add_argument("--sync-delete", action="store_true",
                        help="Delete vector docs not present in SQL")
    args = parser.parse_args()

    out = ingest_from_sqlserver(
        view=args.view,
        where=args.where,
        limit=args.limit,
        batch_size=args.batch_size,
        skip_unchanged=(not args.no_skip_unchanged),
        sync_delete=args.sync_delete,
    )

    print(json.dumps(out, indent=2))

# -------------------------------
# sql/views.sql (example-safe read-only view sketch)
# -------------------------------
-- Example Postgres view. Replace table names/columns to match your schema.
-- Ensure RLS + role grants are configured outside this file.
CREATE OR REPLACE VIEW v_trip_advisor_candidates AS
SELECT
  d.donor_id,
  d.preferred_name AS display_name,
  g.primary_city AS city,
  g.primary_state AS state,
  d.capacity_estimate,
  COALESCE(a.affinity_score, 0) AS affinity_score,
  (SELECT SUM(amount) FROM gifts WHERE gifts.donor_id=d.donor_id AND gifts.date >= CURRENT_DATE - INTERVAL '3 years') AS gifts_3yr,
  (SELECT MAX(date) FROM engagements WHERE engagements.donor_id=d.donor_id) AS last_touch_dt
FROM donor d
JOIN geo g ON g.donor_id = d.donor_id
LEFT JOIN(
  SELECT donor_id, MAX(score) AS affinity_score FROM affinity_signals GROUP BY donor_id
) a ON a.donor_id = d.donor_id;

-- Example export query(CSV)
-- \copy (
--   SELECT donor_id, display_name, city, state, capacity_estimate, affinity_score, gifts_3yr, last_touch_dt,
--          ARRAY_TO_STRING(interests, '|') AS interests
--   FROM v_trip_advisor_candidates
-- ) TO 'exports/donor_profiles.csv' WITH CSV HEADER;

# -------------------------------
# README.md (quickstart)
# -------------------------------
# RAG Fundraising Trip Advisor — Minimal Scaffold

## 1) Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure
Create `.env` or export env vars:
```
OPENAI_API_KEY=sk-...
STORAGE_BACKEND=chroma
STORAGE_PATH=./storage/local
```

## 3) Ingest example data
```
python ingest.py ./exports/donor_profiles.csv
```

## 4) Run API
```
uvicorn app:app --reload --port 8000
```

## 5) Try endpoints
- POST http://localhost:8000/build-itinerary
```
{
  "city": "Austin",
  "dates": ["2025-10-13", "2025-10-14"],
  "query": "Engineering scholarship interest; lapsed in last 18 months; high upgrade potential",
  "filters": {"city": "Austin", "capacity_bucket": [3,4,5]}
}
```

## Run with a local Ollama model (chat optional)
- Ensure Ollama is running: `ollama serve`
- (Optional) Pull a local embedding model: `ollama pull nomic-embed-text`
- Set env vars:
```
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_CHAT_MODEL=gpt-loss:8b   # set to your installed model name
```

## New: Get the top 5 prospects (no itinerary)
Start the API:
```
uvicorn app:app --reload --port 8000
```
Call the endpoint (example for Austin, TX):
```
curl -s -X POST http://127.0.0.1:8000/top5 \
  -H "Content-Type: application/json" \
  -d '{"state":"TX","city":"Austin"}' | jq .
```
Custom WHERE clause:
```
curl -s -X POST http://127.0.0.1:8000/top5 \
  -H "Content-Type: application/json" \
  -d '{"where":"prospect_tier IN (\'Tier 1 - Principal\', \'Tier 2 - Major\') AND state = \'TX\'"}' | jq .
```

## Notes
- This scaffold stores **only aggregated donor signals**. Keep PII (emails/phones/addresses) **out** of the vector index.
- Swap `VectorStore` to pgvector in `storage/vector_store.py` if you prefer Postgres.
- Extend `/top5` and `/search` with additional filters (e.g., `prospect_tier`, `segment`).
