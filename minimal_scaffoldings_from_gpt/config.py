import os
from pydantic import BaseModel


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
