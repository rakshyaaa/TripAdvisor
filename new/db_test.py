import pyodbc

print(pyodbc.drivers())

MSSQL = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=advancementreporting.win.louisiana.edu;"
    "DATABASE=CRM_Advance;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

SQL = """
SELECT * 
FROM [CRM_Advance].[dbo].[view_wealth_engine_prospect_scores]
"""

cn = pyodbc.connect(MSSQL)
cur = cn.cursor()
cur.execute(SQL)

for row in cur.fetchall():
    print(row)
