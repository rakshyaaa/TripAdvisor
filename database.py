import os
import pyodbc
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv


load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

connection_uri = (
    f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)

db = SQLDatabase.from_uri(connection_uri)


db_tool = QuerySQLDataBaseTool(db=db)

# print(db.run("select * from stakeholder_meetings"))


def recommend_next_trip(stakeholder_name: str):
    query = f"""
    SELECT organization, city, country, engagement_score, title, position, last_meeting_date, next_steps
    FROM stakeholder_meetings
    WHERE name = 'Alice Johnson'
    ORDER BY engagement_score DESC, last_meeting_date ASC;
    """
    results = db.run(query)
    return f"Recommended next visits for {stakeholder_name}: {results}"
