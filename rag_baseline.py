from toy_corpus import DOCS

def naive_vector_search(query: str, top_k: int = 3):
    # Intentionally dumb “retrieval”: keyword overlap score
    q = set(query.lower().split())
    scored = []
    for d in DOCS:
        t = set(d["text"].lower().split())
        score = len(q.intersection(t))
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k]]

def naive_generate_answer(query: str, contexts):
    # Fake “LLM”: just concatenates and pretends to answer
    joined = "\n".join([f"- ({c['id']}) {c['text']}" for c in contexts])
    return f"Q: {query}\n\nRetrieved:\n{joined}\n\nAnswer:\nUse SSL certificates (PEM/DER) and rotate them regularly."

if __name__ == "__main__":
    query = "How do I configure SSL certificates for production deployment?"
    ctx = naive_vector_search(query, top_k=3)
    print(naive_generate_answer(query, ctx))
