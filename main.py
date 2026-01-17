# demo.py
# One-file, runnable lecture demo for: baseline RAG -> self-corrective -> HyDE -> multi-hop -> evaluation
# Run: python demo.py

from dataclasses import dataclass
from typing import List, Dict, Tuple

# -----------------------------
# Tiny "documentation corpus"
# -----------------------------
DOCS = [
    {
        "id": "ssl_formats",
        "text": "SSL certificates are commonly encoded in PEM or DER formats. PEM files often contain a certificate chain.",
    },
    {
        "id": "dev_setup_ssl_mention",
        "text": "For local development, you can run the app on http://localhost:3000. Some teams mention SSL in dev, but it is not required.",
    },
    {
        "id": "rotation_policy",
        "text": "Certificate rotation policy: rotate every 90 days. Automate renewal and monitor expiry dates.",
    },
    {
        "id": "prod_ssl_steps",
        "text": "Production SSL setup steps: 1) Put fullchain.pem and privkey.pem in /etc/myapp/tls/. "
        "2) Set TLS_CERT_PATH and TLS_KEY_PATH. 3) Set HTTPS_PORT=443. 4) Restart service.",
    },
    {
        "id": "container_port_note",
        "text": "For containerized deployments behind a reverse proxy, expose 8443 internally and terminate TLS at the proxy. "
        "Do not bind 443 in the container.",
    },
    {
        "id": "compliance_req",
        "text": "Compliance requires TLS 1.2+ and disallows weak ciphers. All production endpoints must support modern cipher suites.",
    },
    {
        "id": "infra_constraint",
        "text": "Infrastructure constraint: only the reverse proxy can bind privileged ports (e.g., 443). "
        "Application containers must use high ports.",
    },
]

# -----------------------------
# Helpers: printing
# -----------------------------
def hr(title: str = "", ch: str = "-", width: int = 86) -> None:
    if title:
        pad = max(width - len(title) - 2, 0)
        left = pad // 2
        right = pad - left
        print(f"\n{ch * left} {title} {ch * right}")
    else:
        print(f"\n{ch * width}")

def show_contexts(contexts: List[Dict]) -> None:
    for c in contexts:
        print(f"- ({c['id']}) {c['text']}")

# -----------------------------
# "Vector search" (intentionally dumb)
# -----------------------------
def naive_vector_search(query: str, top_k: int = 3) -> List[Dict]:
    # Keyword overlap score. This is intentionally brittle to illustrate failure modes.
    q = set(tokenize(query))
    scored: List[Tuple[int, Dict]] = []
    for d in DOCS:
        t = set(tokenize(d["text"]))
        score = len(q.intersection(t))
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k]]

def tokenize(text: str) -> List[str]:
    return [w.strip("?,.!:;()[]{}\"'").lower() for w in text.split() if w.strip("?,.!:;()[]{}\"'")]

# -----------------------------
# Baseline "generation"
# -----------------------------
def baseline_generate_answer(query: str, contexts: List[Dict]) -> str:
    # Fake "LLM": produces plausible-sounding but ungrounded generic advice.
    # This is meant to show why you need faithfulness checks.
    return (
        "Use SSL certificates in PEM/DER format and ensure you rotate them regularly. "
        "Enable modern TLS and configure the HTTPS port appropriately."
    )

# -----------------------------
# Self-corrective RAG (reflection + reformulation)
# -----------------------------
def reflect(question: str, contexts: List[Dict]) -> Dict[str, str]:
    # Deterministic stand-in for "LLM judge": checks whether step-by-step production config is present.
    text = " ".join([c["text"].lower() for c in contexts])
    must_have = ["production ssl setup steps", "fullchain.pem", "privkey.pem", "tls_cert_path", "tls_key_path"]
    has_steps = any(m in text for m in must_have)
    if has_steps:
        return {"decision": "YES", "why": "Retrieved context contains explicit production setup steps."}
    return {
        "decision": "NO",
        "why": "Context is related to SSL but lacks concrete production configuration steps (files/paths/env vars/ports).",
    }

def reformulate(question: str, reflection_why: str) -> str:
    # Simple query expansion guided by what's missing.
    return question + " step-by-step setup fullchain.pem privkey.pem TLS_CERT_PATH TLS_KEY_PATH HTTPS_PORT"

def grounded_answer_from_context(question: str, contexts: List[Dict]) -> str:
    # "Answer" by selecting the most appropriate chunk if present.
    for c in contexts:
        if c["id"] == "prod_ssl_steps":
            return c["text"]
    return "Answer may be incomplete: I couldn't find explicit production setup steps in retrieved context."

def self_corrective_rag(question: str, max_iters: int = 2, top_k: int = 3) -> Tuple[List[Dict], str]:
    q = question
    best_ctx: List[Dict] = []
    for i in range(max_iters):
        ctx = naive_vector_search(q, top_k=top_k)
        best_ctx = ctx
        r = reflect(question, ctx)
        print(f"\nReflection (iter {i+1}): {r['decision']} — {r['why']}")
        if r["decision"] == "YES":
            return ctx, grounded_answer_from_context(question, ctx)
        q = reformulate(q, r["why"])
        print(f"Reformulated query: {q}")
    return best_ctx, grounded_answer_from_context(question, best_ctx)

# -----------------------------
# HyDE (hypothetical doc embedding)
# -----------------------------
def generate_hypothetical_doc(query: str) -> str:
    # Deterministic stand-in for "LLM writes a doc-style answer"
    q = query.lower()
    if "crashing" in q and "startup" in q:
        return (
            "Application startup failures often occur due to missing configuration files, "
            "incorrect database credentials, or missing required environment variables."
        )
    if "ssl" in q and ("configure" in q or "certificates" in q or "production" in q):
        return (
            "Production SSL configuration typically involves placing certificate files "
            "(fullchain.pem, privkey.pem) on disk, setting certificate and key paths via environment "
            "variables, and binding HTTPS to port 443 or terminating TLS at a reverse proxy."
        )
    if "error" in q and len(q.split()) > 12:
        # A nod to the "HyDE can be unhelpful for detailed queries" point.
        return "This query already contains detailed technical text; retrieval should match specifics rather than a generic explanation."
    return "Technical documentation that answers the query in declarative form."

def hyde_retrieve(query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
    hypo = generate_hypothetical_doc(query)
    ctx = naive_vector_search(hypo, top_k=top_k)
    return hypo, ctx

# -----------------------------
# Multi-hop reasoning demo
# -----------------------------
def naive_multihop_answer(question: str, contexts: List[Dict]) -> str:
    # Intentionally wrong synthesis: ignores constraints.
    return "Use port 443 everywhere and enable the newest TLS settings."

def constraint_targeted_retrieve(question: str, top_k: int = 4) -> List[Dict]:
    # Cheap hack: explicitly add constraint keywords to push retrieval to constraint docs.
    q = question + " compliance requires infrastructure constraint reverse proxy privileged ports cipher suites TLS 1.2"
    return naive_vector_search(q, top_k=top_k)

def grounded_multihop_answer(question: str, contexts: List[Dict]) -> str:
    # Deterministic synthesis based on expected constraint chunks.
    ids = {c["id"] for c in contexts}
    lines = []
    if "infra_constraint" in ids or "container_port_note" in ids:
        lines.append("- Terminate TLS at the reverse proxy; app containers should not bind 443.")
        lines.append("- Expose a high port internally (e.g., 8443) behind the proxy.")
    if "compliance_req" in ids:
        lines.append("- Enforce TLS 1.2+ and modern cipher suites to satisfy compliance.")
    if not lines:
        lines.append("- Constraints not found in context; answer would be risky.")

    evidence = "\n".join([f"  * ({c['id']}) {c['text']}" for c in contexts])
    return "Grounded synthesis:\n" + "\n".join(lines) + "\n\nEvidence:\n" + evidence

# -----------------------------
# Simple evaluation (RAGAS-like shape, deterministic)
# -----------------------------
def metric_context_relevance(question: str, contexts: List[Dict]) -> float:
    q = set(tokenize(question))
    good = 0
    for c in contexts:
        t = c["text"].lower()
        if any(w in t for w in q if w not in {"the", "a", "an", "do", "i", "how", "to", "is", "are", "for", "of", "and"}):
            good += 1
    return good / max(len(contexts), 1)

def metric_faithfulness(answer: str, contexts: List[Dict]) -> float:
    # Very rough: checks whether answer contains terms present in context.
    ctx_text = " ".join([c["text"].lower() for c in contexts])
    claims = [s.strip().lower() for s in answer.split(".") if s.strip()]
    if not claims:
        return 0.0
    supported = 0
    for cl in claims:
        toks = [t for t in tokenize(cl) if len(t) > 3]
        if toks and any(tok in ctx_text for tok in toks):
            supported += 1
    return supported / len(claims)

def metric_answer_relevance(question: str, answer: str) -> float:
    q = set([w for w in tokenize(question) if w not in {"the", "a", "an", "do", "i", "how", "to", "is", "are", "for", "of", "and"}])
    a = set(tokenize(answer))
    return len(q.intersection(a)) / max(len(q), 1)

def evaluate(question: str, contexts: List[Dict], answer: str) -> Dict[str, float]:
    return {
        "context_relevance": round(metric_context_relevance(question, contexts), 3),
        "faithfulness": round(metric_faithfulness(answer, contexts), 3),
        "answer_relevance": round(metric_answer_relevance(question, answer), 3),
    }

def print_scores(scores: Dict[str, float]) -> None:
    print("Scores:")
    for k, v in scores.items():
        print(f"  {k}: {v}")

# -----------------------------
# Demo Runner
# -----------------------------
@dataclass
class DemoCase:
    title: str
    question: str

def run_baseline(case: DemoCase, top_k: int = 3) -> None:
    hr(f"BASELINE RAG (expected to fail): {case.title}")
    print("Question:", case.question)

    ctx = naive_vector_search(case.question, top_k=top_k)
    print("\nRetrieved context:")
    show_contexts(ctx)

    ans = baseline_generate_answer(case.question, ctx)
    print("\nGenerated answer:")
    print(ans)

    print()
    print_scores(evaluate(case.question, ctx, ans))

def run_self_corrective(case: DemoCase, top_k: int = 3) -> None:
    hr(f"SELF-CORRECTIVE RAG: {case.title}")
    print("Question:", case.question)

    ctx, ans = self_corrective_rag(case.question, max_iters=2, top_k=top_k)

    print("\nFinal retrieved context:")
    show_contexts(ctx)

    print("\nGenerated answer (grounded):")
    print(ans)

    print()
    print_scores(evaluate(case.question, ctx, ans))

def run_hyde(case: DemoCase, top_k: int = 3) -> None:
    hr(f"HyDE RETRIEVAL: {case.title}")
    print("Question:", case.question)

    hypo, ctx = hyde_retrieve(case.question, top_k=top_k)

    print("\nHyDE hypothetical document:")
    print(hypo)

    print("\nRetrieved context:")
    show_contexts(ctx)

    ans = grounded_answer_from_context(case.question, ctx)
    print("\nGenerated answer (attempt grounded):")
    print(ans)

    print()
    print_scores(evaluate(case.question, ctx, ans))

def run_multihop(case: DemoCase, top_k_naive: int = 3, top_k_better: int = 4) -> None:
    hr(f"MULTI-HOP REASONING: {case.title}")
    print("Question:", case.question)

    ctx_bad = naive_vector_search(case.question, top_k=top_k_naive)
    ans_bad = naive_multihop_answer(case.question, ctx_bad)
    print("\nNaive retrieval IDs:", [c["id"] for c in ctx_bad])
    print("Naive answer:")
    print(ans_bad)
    print()
    print_scores(evaluate(case.question, ctx_bad, ans_bad))

    ctx_good = constraint_targeted_retrieve(case.question, top_k=top_k_better)
    ans_good = grounded_multihop_answer(case.question, ctx_good)
    print("\nImproved retrieval IDs:", [c["id"] for c in ctx_good])
    print("Grounded synthesis answer:")
    print(ans_good)
    print()
    print_scores(evaluate(case.question, ctx_good, ans_good))

def main():
    # Case 1: mirrors your "SSL production setup" failure mode
    ssl_case = DemoCase(
        title="Retrieval precision + coherence (SSL in production)",
        question="How do I configure SSL certificates for production deployment?",
    )

    # Case 2: HyDE shines with short/vague questions vs doc-style answers
    hyde_case_short = DemoCase(
        title="HyDE helps short question match doc language",
        question="Why is my application crashing on startup?",
    )

    # Case 3: HyDE doesn't help as much with very detailed queries
    hyde_case_long = DemoCase(
        title="HyDE is less useful for already-detailed queries",
        question="My app crashes on startup with error: 'missing required database connection parameters in config.yaml'. What should I check?",
    )

    # Case 4: Multi-hop constraints
    multihop_case = DemoCase(
        title="Multi-hop constraints (compliance + infra)",
        question="Given our compliance requirements and infrastructure constraints, what SSL configuration should we use?",
    )

    # Run in a lecture-friendly order: fail -> fix -> alternate fix -> reason -> measure
    run_baseline(ssl_case)
    run_self_corrective(ssl_case)

    run_hyde(hyde_case_short)
    run_hyde(hyde_case_long)

    run_multihop(multihop_case)

    hr("DONE", ch="=")

if __name__ == "__main__":
    main()
