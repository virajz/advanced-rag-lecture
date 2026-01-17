from rag_baseline import naive_vector_search

def naive_answer(question: str, ctx):
    # Shows what happens when you miss constraint docs
    ids = [c["id"] for c in ctx]
    return f"Retrieved: {ids}\nAnswer: Use port 443 and enable TLS 1.3."

def better_retrieve_for_constraints(question: str):
    # A cheap trick: add constraint keywords explicitly
    q = question + " compliance requires infrastructure constraint reverse proxy privileged ports"
    return naive_vector_search(q, top_k=4)

def grounded_answer(question: str, ctx):
    text = " ".join([c["text"] for c in ctx])
    return ("Grounded synthesis:\n"
            "- Terminate TLS at the reverse proxy (only it can bind 443).\n"
            "- Containers expose a high port (e.g., 8443) internally.\n"
            "- Ensure TLS 1.2+ and modern cipher suites to satisfy compliance.\n\n"
            f"Evidence excerpts:\n{text}")

if __name__ == "__main__":
    q = "Given our compliance requirements and infrastructure constraints, what SSL configuration should we use?"
    ctx_bad = naive_vector_search(q, top_k=3)
    print("NAIVE:\n", naive_answer(q, ctx_bad))

    ctx_good = better_retrieve_for_constraints(q)
    print("\nIMPROVED RETRIEVAL IDs:", [c["id"] for c in ctx_good])
    print("\nBETTER:\n", grounded_answer(q, ctx_good))
