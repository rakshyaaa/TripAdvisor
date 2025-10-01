
from ollama import Client as OllamaClient
from rag import retrieve, build_prompt

ollama = OllamaClient(host="http://localhost:11434")


def answer_with_ollama(question: str, city: str = None, weps_range=None, k: int = 5) -> str:
    hits = retrieve(question, k=k, city=city, weps_range=weps_range)
    # print("Hits: ", hits)  # DEBUG
    # user query and retrived context from chromadb
    prompt = build_prompt(question, hits)
    # print("Prompt: ", prompt)  # DEBUG
    resp = ollama.chat(model="gpt-oss:20b", messages=[
                       {"role": "user", "content": prompt}])
    # print("Response: ", resp)  # DEBUG
    return resp["message"]["content"]


if __name__ == "__main__":
    q = "Top prospects with WEPS greater than 80. Give me their names and their primary manager"
    print(answer_with_ollama(q, city=None, weps_range=None, k=5))
