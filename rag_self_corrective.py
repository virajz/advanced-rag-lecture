from toy_corpus import DOCS
from rag_baseline import naive_vector_search

def reflect(question: str, contexts) -> dict:
    # Stand-in for an LLM judge. Deterministic rules for the demo.
    text = " ".join([c["text"].lower() for c in contexts])
    must_have = ["production ssl setup steps", "fullchain.pem", "privkey.pem", "tls_cert_path", "tls_key_path"]
    has_steps = any(m in text for m in must_have)
    if has_steps:
        return {"decision": "YES", "why": "Retrieved context contains explicit production setup steps."}
    return {
        "decision": "NO",
        "why": "Context talks about SSL generally but lacks step-by-step production configuration (paths/env vars/ports)."
    }

def reformulate(question: str, reflection_why: str) -> str:
    # Simulate query expansion guided by the reflection
    return question + " step-by-step setup fullchain.pem privkey.pem TLS_CERT_PATH TLS_KEY_PATH"

def generate_answer_from_context(question: str, contexts) -> str:
    # “Answer” by extracting the steps chunk if present
    for c in contexts:
        if c["id"] == "prod_ssl_steps":
            return f"Answer grounded in docs:\n{c['text']}"
    return "Answer may be incomplete: I couldn't find explicit production setup steps in retrieved context."

def self_corrective_rag(question: str, max_iters: int = 2):
    q = question
    best_ctx = None
    for i in range(max_iters):
        ctx = naive_vector_search(q, top_k=3)
        best_ctx = ctx
        r = reflect(question, ctx)
        print(f"\nIteration {i+1} reflection: {r['decision']} — {r['why']}")
        if r["decision"] == "YES":
            return generate_answer_from_context(question, ctx)
        q = reformulate(q, r["why"])
        print(f"Reformulated query: {q}")
    return generate_answer_from_context(question, best_ctx)

if __name__ == "__main__":
    question = "How do I configure SSL certificates for production deployment?"
    print(self_corrective_rag(question))
