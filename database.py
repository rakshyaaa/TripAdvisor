import os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from dotenv import load_dotenv


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
