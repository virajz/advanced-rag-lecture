def context_relevance(question: str, contexts) -> float:
    # Counts how many chunks contain tokens that look like they address the question
    q = set([w.strip("?,.").lower() for w in question.split()])
    good = 0
    for c in contexts:
        t = c["text"].lower()
        if any(w in t for w in q):
            good += 1
    return good / max(len(contexts), 1)

def faithfulness(answer: str, contexts) -> float:
    # Very rough: penalize claims not in context
    ctx_text = " ".join([c["text"].lower() for c in contexts])
    claims = [s.strip().lower() for s in answer.split(".") if s.strip()]
    supported = sum(1 for cl in claims if any(tok in ctx_text for tok in cl.split()))
    return supported / max(len(claims), 1)

def answer_relevance(question: str, answer: str) -> float:
    # Rough overlap score
    q = set([w.strip("?,.").lower() for w in question.split()])
    a = set([w.strip("?,.").lower() for w in answer.split()])
    return len(q.intersection(a)) / max(len(q), 1)

def evaluate(question: str, contexts, answer: str):
    return {
        "context_relevance": context_relevance(question, contexts),
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevance": answer_relevance(question, answer),
    }
