from typing import List
from config import Settings

settings = Settings()


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
