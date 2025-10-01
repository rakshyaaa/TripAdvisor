# offline_index_tables.py
# Build a table/view index: summarize → embed → store in Chroma
# Usage:
#   export MSSQL_ODBC_CONNECT="Driver={ODBC Driver 18 for SQL Server};Server=tcp:YOURSERVER,1433;Database=CRM_Advance;UID=USER;PWD=PASS;Encrypt=yes;TrustServerCertificate=no"
#   export OLLAMA_BASE_URL="http://127.0.0.1:11434"
#   python offline_index_tables.py --allow dbo.view_wealth_engine_prospect_scores --storage ./table_index

import os
import argparse
import requests
import pandas as pd
import chromadb
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


def sql_engine(odbc):
    return create_engine("mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc), pool_pre_ping=True)


def list_columns(engine, table):
    q = text("""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t
        ORDER BY ORDINAL_POSITION
    """)
    schema, name = table.split(".", 1)[0], table.split(".", 1)[1]
    # handle dbo.view_name style
    if "." in name:
        schema, name = table.split(".", 1)
        if "." in name:
            schema, name = table.split(".", 1)  # best-effort
    schema = table.split(".")[0]
    name = table.split(".")[1]
    with engine.begin() as cxn:
        rows = cxn.execute(q, {"s": schema, "t": name}).fetchall()
    return [(r.COLUMN_NAME, r.DATA_TYPE) for r in rows]


def sample_rows(engine, table, n=5):
    with engine.begin() as cxn:
        df = pd.read_sql(text(f"SELECT TOP ({n}) * FROM {table}"), cxn)
    return df


def ollama_generate(prompt, model="gpt-loss:8b", base="http://127.0.0.1:11434"):
    r = requests.post(f"{base.rstrip('/')}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def ollama_embed(text, model="nomic-embed-text", base="http://127.0.0.1:11434"):
    r = requests.post(f"{base.rstrip('/')}/api/embeddings",
                      json={"model": model, "input": text}, timeout=120)
    if r.status_code != 200:
        r = requests.post(f"{base.rstrip('/')}/api/embeddings",
                          json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json().get("embedding") or r.json().get("data", [{}])[0].get("embedding")


SUMMARY_PROMPT = """You are documenting a SQL object for a Text-to-SQL assistant.

TABLE/VIEW NAME: {name}

COLUMNS:
{columns}

SAMPLE ROWS (may be truncated):
{samples}

Write a concise 4-8 sentence summary describing:
- what this table/view represents,
- the most important columns and how they’re used,
- typical filters (e.g., city, state, tiers),
- typical sorts (e.g., prospect_potential_score DESC),
- any safety notes (read-only, aggregate scores, no PII).

Return plain text (no markdown)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", nargs="+", required=True,
                    help="Allowed tables/views, e.g. dbo.view_wealth_engine_prospect_scores")
    ap.add_argument("--odbc", default=os.getenv("MSSQL_ODBC_CONNECT"))
    ap.add_argument(
        "--ollama", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--chat-model", default=os.getenv("OLLAMA_CHAT_MODEL", "gpt-loss:8b"))
    ap.add_argument("--embed-model",
                    default=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    ap.add_argument("--storage", default="./table_index")
    ap.add_argument("--collection", default="table_summaries")
    args = ap.parse_args()
    if not args.odbc:
        raise SystemExit("Set MSSQL_ODBC_CONNECT or pass --odbc")

    eng = sql_engine(args.odbc)
    client = chromadb.PersistentClient(path=args.storage)
    coll = client.get_or_create_collection(args.collection)

    for tbl in args.allow:
        cols = list_columns(eng, tbl)
        samp = sample_rows(eng, tbl, n=5)
        col_txt = "\n".join([f"- {c} ({t})" for c, t in cols])
        samp_txt = samp.head(5).to_csv(
            index=False) if not samp.empty else "(no rows)"
        prompt = SUMMARY_PROMPT.format(
            name=tbl, columns=col_txt, samples=samp_txt)
        summary = ollama_generate(
            prompt, model=args.chat_model, base=args.ollama)
        emb = ollama_embed(summary, model=args.embed_model, base=args.ollama)

        coll.upsert(
            ids=[tbl],
            documents=[summary],
            metadatas=[{"table": tbl, "columns": [c for c, _ in cols]}],
            embeddings=[emb],
        )
        print(f"Indexed: {tbl}")

    print("Done.")


if __name__ == "__main__":
    main()
