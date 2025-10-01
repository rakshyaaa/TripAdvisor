import pandas as pd
import re
import argparse
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from config import Settings
from vector_sore import VectorStore, VectorDoc
from retriever import upsert_donor_docs
from dataclasses import dataclass
from typing import List, Dict, Any


VIEW_DEFAULT = "dbo.view_wealth_engine_prospect_scores"

DELIMS = re.compile(r"[;,|/]+")


def _capacity_estimate_from_band(band: str | None) -> int:
    if not band:
        return 3
    b = band.strip().upper()
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
    uniq = []
    for x in out:
        if x.lower() not in seen:
            seen.add(x.lower())
            uniq.append(x)
    return uniq


def _row_to_doc(r: pd.Series) -> dict:
    # Map SQL view → donor doc used by the RAG store
    capacity_estimate = _capacity_estimate_from_band(r.get("capacity_band"))
    affinity_score = float(r.get("propensity_score") or 0.0) / 100.0
    lifetime_giving = float(r.get("lifetime_giving") or 0.0)
    interests = _interests_from_inclination(
        r.get("inclinationAffiliation"), r.get("boardMembership"))

    return {
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


def ingest_from_sqlserver(view: str = VIEW_DEFAULT, where: str | None = None, limit: int | None = None):
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

    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    upsert_donor_docs(docs)
    return {"rows": len(docs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest prospects from SQL Server view into the vector store")
    parser.add_argument("--view", default=VIEW_DEFAULT,
                        help="Qualified view name")
    parser.add_argument("--where", default=None,
                        help="Optional WHERE clause, e.g. state = 'TX' AND city = 'Austin'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional row limit (uses OFFSET/FETCH)")
    args = parser.parse_args()

    out = ingest_from_sqlserver(
        view=args.view, where=args.where, limit=args.limit)
    print(f"Ingested {out['rows']} rows from {args.view}")
