from toy_corpus import DOCS
from rag_baseline import naive_vector_search

def generate_hypothetical_doc(query: str) -> str:
    # Deterministic stand-in for “LLM writes doc-style answer”
    if "crashing" in query.lower() and "startup" in query.lower():
        return ("Application startup failures often occur due to missing configuration files, "
                "incorrect database credentials, or missing required environment variables.")
    if "configure ssl" in query.lower() or "ssl certificates" in query.lower():
        return ("Production SSL configuration typically involves placing certificate files "
                "(fullchain.pem, privkey.pem) on disk, setting certificate/key paths via environment "
                "variables, and binding HTTPS to port 443 or proxy-terminated TLS.")
    return "General technical documentation answering the query in declarative form."

def hyde_retrieve(query: str, top_k: int = 3):
    hypo = generate_hypothetical_doc(query)
    print("\nHyDE hypothetical document:\n", hypo)
    return naive_vector_search(hypo, top_k=top_k)

if __name__ == "__main__":
    q1 = "Why is my application crashing on startup?"
    ctx1 = hyde_retrieve(q1)
    print("\nRetrieved IDs:", [c["id"] for c in ctx1])

    q2 = "How do I configure SSL certificates for production deployment?"
    ctx2 = hyde_retrieve(q2)
    print("\nRetrieved IDs:", [c["id"] for c in ctx2])
