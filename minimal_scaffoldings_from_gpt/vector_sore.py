from typing import List
from config import Settings
from typing import List, Dict, Any


settings = Settings()


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
