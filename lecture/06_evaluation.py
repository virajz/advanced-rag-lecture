"""
Step 6: RAG Evaluation Metrics
===============================

How do you know if your RAG system is actually good?
You need METRICS - not just "it feels right".

This script implements the three core RAG evaluation metrics:
1. Context Relevance - Did we retrieve the RIGHT documents?
2. Faithfulness - Is the answer grounded in the context (no hallucination)?
3. Answer Relevance - Does the answer address the question?

These are the same metrics used by RAGAS (RAG Assessment) framework.

Run: uv run python lecture/06_evaluation.py
"""

import os
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one API call."""
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================================================
# EVALUATION METRICS (LLM-as-Judge)
# =============================================================================

def evaluate_context_relevance(question: str, contexts: list[dict]) -> dict:
    """
    CONTEXT RELEVANCE: Are the retrieved documents relevant to the question?

    For each context, ask the LLM: "Is this useful for answering the question?"
    Score = (number of relevant contexts) / (total contexts)

    High score = Good retrieval
    Low score = Retrieved irrelevant documents (wasted context window)
    """
    if not contexts:
        return {"score": 0.0, "details": []}

    details = []

    for ctx in contexts:
        prompt = f"""You are evaluating whether a document is relevant for answering a question.

QUESTION: {question}

DOCUMENT: {ctx['text']}

Is this document useful for answering the question? Consider:
- Does it contain information related to the question?
- Would this help formulate an answer?

Respond with ONLY "YES" or "NO"."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        is_relevant = "YES" in response.choices[0].message.content.upper()
        details.append({
            "id": ctx["id"],
            "relevant": is_relevant
        })

    relevant_count = sum(1 for d in details if d["relevant"])
    score = relevant_count / len(contexts)

    return {"score": score, "details": details}


def evaluate_faithfulness(answer: str, contexts: list[dict]) -> dict:
    """
    FAITHFULNESS: Is the answer grounded in the retrieved context?

    Extract claims from the answer, then check if each claim is supported
    by the context. Catches hallucination!

    Score = (supported claims) / (total claims)

    High score = Answer is grounded in documents
    Low score = Answer contains hallucinated information
    """
    # Step 1: Extract claims from the answer
    extract_prompt = f"""Extract the key factual claims from this answer as a numbered list.
Each claim should be a single, verifiable statement.
If there are no factual claims, respond with "NO CLAIMS".

ANSWER: {answer}

CLAIMS:"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": extract_prompt}]
    )

    claims_text = response.choices[0].message.content

    if "NO CLAIMS" in claims_text.upper():
        return {"score": 1.0, "claims": [], "details": "No factual claims to verify"}

    # Parse claims (simple line-by-line)
    claims = [
        line.strip().lstrip("0123456789.-) ")
        for line in claims_text.strip().split("\n")
        if line.strip() and not line.strip().startswith("CLAIMS")
    ]

    if not claims:
        return {"score": 1.0, "claims": [], "details": "No claims extracted"}

    # Step 2: Check each claim against context
    context_text = "\n".join([ctx["text"] for ctx in contexts])
    details = []

    for claim in claims:
        verify_prompt = f"""You are checking if a claim is supported by the given context.

CONTEXT:
{context_text}

CLAIM: {claim}

Is this claim supported by (can be inferred from) the context?
Respond with ONLY "SUPPORTED" or "NOT SUPPORTED"."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": verify_prompt}]
        )

        is_supported = "SUPPORTED" in response.choices[0].message.content.upper() and \
                       "NOT SUPPORTED" not in response.choices[0].message.content.upper()
        details.append({
            "claim": claim,
            "supported": is_supported
        })

    supported_count = sum(1 for d in details if d["supported"])
    score = supported_count / len(claims) if claims else 1.0

    return {"score": score, "claims": claims, "details": details}


def evaluate_answer_relevance(question: str, answer: str) -> dict:
    """
    ANSWER RELEVANCE: Does the answer actually address the question?

    You can have high faithfulness but low relevance if the answer
    is accurate but doesn't address what was asked.

    Uses LLM to judge if the answer addresses the question.
    Also uses embedding similarity as a secondary signal.
    """
    # LLM judgment
    prompt = f"""You are evaluating whether an answer addresses a question.

QUESTION: {question}

ANSWER: {answer}

Rate how well the answer addresses the question:
- FULLY: The answer directly and completely addresses the question
- PARTIALLY: The answer addresses some aspects but misses key parts
- NOT: The answer does not address the question at all

Respond with ONLY one word: FULLY, PARTIALLY, or NOT."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    judgment = response.choices[0].message.content.strip().upper()

    if "FULLY" in judgment:
        llm_score = 1.0
    elif "PARTIALLY" in judgment:
        llm_score = 0.5
    else:
        llm_score = 0.0

    # Embedding similarity as secondary signal
    embeddings = get_embeddings_batch([question, answer])
    embedding_similarity = cosine_similarity(embeddings[0], embeddings[1])

    # Combine scores (weighted average)
    combined_score = 0.7 * llm_score + 0.3 * embedding_similarity

    return {
        "score": combined_score,
        "llm_judgment": judgment,
        "llm_score": llm_score,
        "embedding_similarity": embedding_similarity
    }


def evaluate_rag_response(question: str, contexts: list[dict], answer: str) -> dict:
    """
    Complete RAG evaluation combining all three metrics.
    """
    context_relevance = evaluate_context_relevance(question, contexts)
    faithfulness = evaluate_faithfulness(answer, contexts)
    answer_relevance = evaluate_answer_relevance(question, answer)

    return {
        "context_relevance": context_relevance,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "summary": {
            "context_relevance": context_relevance["score"],
            "faithfulness": faithfulness["score"],
            "answer_relevance": answer_relevance["score"]
        }
    }


# =============================================================================
# SIMPLE VECTOR STORE & RAG (for testing)
# =============================================================================

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs: list[dict]):
        texts = [doc["text"] for doc in docs]
        embeddings = get_embeddings_batch(texts)
        for doc, embedding in zip(docs, embeddings):
            self.documents.append(doc)
            self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = get_embeddings_batch([query])[0]
        results = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append({"id": doc["id"], "text": doc["text"], "similarity": similarity})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


def generate_answer(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join([f"[{ctx['id']}]: {ctx['text']}" for ctx in contexts])
    prompt = f"""Answer the question based ONLY on the provided context.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# =============================================================================
# DEMONSTRATION
# =============================================================================

def print_evaluation(eval_result: dict):
    """Pretty print evaluation results."""
    print("\n--- EVALUATION SCORES ---")
    print(f"  Context Relevance: {eval_result['summary']['context_relevance']:.2f}")
    print(f"  Faithfulness:      {eval_result['summary']['faithfulness']:.2f}")
    print(f"  Answer Relevance:  {eval_result['summary']['answer_relevance']:.2f}")

    # Details
    print("\n  Context Relevance Details:")
    for d in eval_result["context_relevance"]["details"]:
        status = "Relevant" if d["relevant"] else "Not Relevant"
        print(f"    [{d['id']}]: {status}")

    print("\n  Faithfulness Details:")
    if isinstance(eval_result["faithfulness"]["details"], list):
        for d in eval_result["faithfulness"]["details"]:
            status = "Supported" if d["supported"] else "NOT Supported"
            print(f"    {status}: {d['claim'][:50]}...")
    else:
        print(f"    {eval_result['faithfulness']['details']}")

    print(f"\n  Answer Relevance: {eval_result['answer_relevance']['llm_judgment']}")


def main():
    print("=" * 60)
    print("RAG Evaluation Metrics")
    print("=" * 60)

    # Build vector store
    print("\n1. Building vector store...")
    vector_store = SimpleVectorStore()
    vector_store.add_documents(DOCS)

    # Test Case 1: Good RAG response
    print("\n" + "=" * 60)
    print("TEST 1: Good Query (should score high)")
    print("=" * 60)

    query1 = "How do I configure SSL certificates for production?"
    contexts1 = vector_store.search(query1, top_k=3)
    answer1 = generate_answer(query1, contexts1)

    print(f"\nQuery: {query1}")
    print(f"\nRetrieved: {[c['id'] for c in contexts1]}")
    print(f"\nAnswer: {answer1[:200]}...")

    eval1 = evaluate_rag_response(query1, contexts1, answer1)
    print_evaluation(eval1)

    # Test Case 2: Query with potential hallucination
    print("\n" + "=" * 60)
    print("TEST 2: Testing Faithfulness (hallucination detection)")
    print("=" * 60)

    query2 = "What port should I use?"
    contexts2 = vector_store.search(query2, top_k=3)

    # Deliberately create an answer with some hallucination
    hallucinated_answer = """Based on the documentation:
1. Use port 443 for HTTPS in production
2. Use port 8443 for containers behind a reverse proxy
3. Use port 8080 for development (this is made up!)
4. Always enable HTTP/3 support (this is also made up!)"""

    print(f"\nQuery: {query2}")
    print(f"\nDeliberately hallucinated answer:\n{hallucinated_answer}")

    eval2 = evaluate_rag_response(query2, contexts2, hallucinated_answer)
    print_evaluation(eval2)

    # Test Case 3: Irrelevant retrieval
    print("\n" + "=" * 60)
    print("TEST 3: Off-topic Query (tests context relevance)")
    print("=" * 60)

    query3 = "How do I make coffee?"
    contexts3 = vector_store.search(query3, top_k=3)
    answer3 = generate_answer(query3, contexts3)

    print(f"\nQuery: {query3}")
    print(f"\nRetrieved: {[c['id'] for c in contexts3]}")
    print(f"\nAnswer: {answer3}")

    eval3 = evaluate_rag_response(query3, contexts3, answer3)
    print_evaluation(eval3)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print("""
┌─────────────────────┬────────┬────────┬────────┐
│ Metric              │ Test 1 │ Test 2 │ Test 3 │
├─────────────────────┼────────┼────────┼────────┤""")
    print(f"│ Context Relevance   │  {eval1['summary']['context_relevance']:.2f}  │  {eval2['summary']['context_relevance']:.2f}  │  {eval3['summary']['context_relevance']:.2f}  │")
    print(f"│ Faithfulness        │  {eval1['summary']['faithfulness']:.2f}  │  {eval2['summary']['faithfulness']:.2f}  │  {eval3['summary']['faithfulness']:.2f}  │")
    print(f"│ Answer Relevance    │  {eval1['summary']['answer_relevance']:.2f}  │  {eval2['summary']['answer_relevance']:.2f}  │  {eval3['summary']['answer_relevance']:.2f}  │")
    print("└─────────────────────┴────────┴────────┴────────┘")

    print("""
KEY TAKEAWAYS:
==============

1. CONTEXT RELEVANCE measures retrieval quality
   - Low score → Improve retrieval (try HyDE, better chunking)

2. FAITHFULNESS measures hallucination
   - Low score → Answer contains unsupported claims
   - Use self-corrective RAG or stricter prompts

3. ANSWER RELEVANCE measures if answer addresses the question
   - Low score → Generation problem, not retrieval

USE THESE METRICS TO:
- Compare baseline vs advanced RAG techniques
- Decide if HyDE/Self-correction is worth the cost
- Monitor production RAG quality over time

NEXT STEPS:
- Implement RAGAS library for production evaluation
- Create test sets for your specific domain
- Set up automated evaluation pipelines
""")


if __name__ == "__main__":
    main()
