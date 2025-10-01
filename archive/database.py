import os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from dotenv import load_dotenv
import pandas as

load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

connection_uri = (
    f"mssql+pyodbc://@{DB_HOST}/{DB_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

db = SQLDatabase.from_uri(connection_uri)


db_tool = QuerySQLDataBaseTool(db=db)


def fetch_candidates():
    query = f"""
    SELECT * from view_wealth_engine_prospect_scores
    """
    results = db.run(query)

    return f"Here is the refined wealth engine data: {results}"


def rank_backend(states=None, cities=None, officer=None, top_n=10) -> Dict[str, Any]:
    params = {
        "States": None if not states else ",".join(states),
        "Cities": None if not cities else ",".join(cities),
        "Officer": officer,
        "TopN": top_n
    }
    sql = text("""
      EXEC dbo.usp_rank_prospects
        @States=:States, @Cities=:Cities, @Officer=:Officer, @TopN=:TopN
    """)
    with engine.begin() as cx:
        df = pd.read_sql(sql, cx, params=params)

    def mk(r):
        reasons = []
        if pd.notna(r.get("wealth_score")):
            reasons.append(f"wealth {int(r['wealth_score'])}")
        if pd.notna(r.get("propensity_score")):
            reasons.append(f"propensity {int(r['propensity_score'])}")
        if pd.notna(r.get("givingCapacity_score")):
            reasons.append(f"capacity {int(r['givingCapacity_score'])}")
        dgl = r.get("days_since_last_gift")
        if pd.notna(dgl) and int(dgl) >= 90:
            reasons.append(f"recent giving {int(dgl)}d ago")
        return {
            "name_or_id": str(r["name_or_id"]),
            "display_name": r.get("display_name") or str(r["name_or_id"]),
            "city": r.get("city"),
            "state": r.get("state"),
            "final_score": float(r["final_score"]),
            "segment": f"{r.get('donor_quadrant')} | {r.get('capacity_band')} | {r.get('prospect_tier')}",
            "reasons": reasons[:3]
        }

    return {
        "filters": {"states": states, "cities": cities, "officer": officer},
        "prospects": [mk(x) for _, x in df.iterrows()]
    }
