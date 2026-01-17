"""
Step 7: RAGAS-Style Automated Evaluation Pipeline
==================================================

This script demonstrates a production-ready evaluation pipeline
inspired by RAGAS (Retrieval Augmented Generation Assessment).

From the lecture:
"RAGAS turns subjective quality assessment into measurable scores.
You define a test set of questions and expected answers, run your
RAG pipeline, and RAGAS computes these metrics automatically."

Key insight: You need metrics that tell you "For this type of query,
self-correction improves answer quality by X percent, which justifies
the Y percent cost increase."

Run: uv run python lecture/07_ragas_evaluation.py
"""

import os
import time
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from mistralai import Mistral

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS
from lecture.vector_store import get_vector_store, get_mistral_embeddings

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def llm_call_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Make LLM call with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise
    return ""


# =============================================================================
# TEST CASE DATA STRUCTURE
# =============================================================================

@dataclass
class RAGTestCase:
    """
    A single test case for RAG evaluation.

    From the lecture: "Create ten test questions that represent
    your typical use cases."
    """
    query: str
    expected_answer: str
    category: str


# Define test cases - reduced to 5 for faster evaluation
TEST_CASES = [
    RAGTestCase(
        query="How do I configure SSL certificates for production?",
        expected_answer="Put fullchain.pem and privkey.pem in /etc/myapp/tls/, set TLS_CERT_PATH and TLS_KEY_PATH, set HTTPS_PORT=443, restart service.",
        category="configuration"
    ),
    RAGTestCase(
        query="What is the certificate rotation policy?",
        expected_answer="Rotate every 90 days. Automate renewal and monitor expiry dates.",
        category="policy"
    ),
    RAGTestCase(
        query="What port should containers use?",
        expected_answer="Use port 8443 internally, terminate TLS at the reverse proxy. Don't bind 443 in containers.",
        category="configuration"
    ),
    RAGTestCase(
        query="What are the TLS compliance requirements?",
        expected_answer="TLS 1.2+ required, weak ciphers disallowed, modern cipher suites required.",
        category="compliance"
    ),
    RAGTestCase(
        query="What SSL formats are supported?",
        expected_answer="PEM and DER formats. PEM files often contain certificate chains.",
        category="reference"
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(vec1, vec2) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def generate_answer(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in contexts])
    prompt = f"""Answer based ONLY on the context. If insufficient, say so.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    return llm_call_with_retry(prompt)


# =============================================================================
# RAGAS-STYLE METRICS (with rate limit handling)
# =============================================================================

def compute_context_relevance(question: str, contexts: list[dict]) -> float:
    """
    From lecture: "For each retrieved chunk, ask the LLM: 'Is this chunk
    useful for answering the question? YES or NO.' Count YES responses
    and divide by total chunks."
    """
    if not contexts:
        return 0.0

    relevant_count = 0
    for ctx in contexts:
        prompt = f"""Is this document useful for answering the question?
QUESTION: {question}
DOCUMENT: {ctx['text']}
Answer only YES or NO."""

        response = llm_call_with_retry(prompt)
        if "YES" in response.upper():
            relevant_count += 1
        time.sleep(0.5)  # Delay between calls to avoid rate limits

    return relevant_count / len(contexts)


def compute_faithfulness(answer: str, contexts: list[dict]) -> float:
    """
    From lecture: "For each claim in the generated answer, can you find
    supporting evidence in the retrieved context?"

    Simplified version: just check if answer content overlaps with context.
    """
    # Simplified: check semantic similarity between answer and combined context
    context_text = " ".join([c["text"] for c in contexts])

    prompt = f"""Does the answer contain only information from the context?
CONTEXT: {context_text}
ANSWER: {answer}

Rate: HIGH (answer is fully grounded), MEDIUM (mostly grounded), LOW (contains unsupported claims)
Respond with only: HIGH, MEDIUM, or LOW"""

    response = llm_call_with_retry(prompt).upper()

    if "HIGH" in response:
        return 1.0
    elif "MEDIUM" in response:
        return 0.7
    return 0.3


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    From lecture: "Does the answer actually address what the user asked?"
    """
    prompt = f"""Rate how well the answer addresses the question.
QUESTION: {question}
ANSWER: {answer}

Rate: FULLY, PARTIALLY, or NOT
Respond with only the rating word."""

    rating = llm_call_with_retry(prompt).upper()

    if "FULLY" in rating:
        return 1.0
    elif "PARTIALLY" in rating:
        return 0.5
    return 0.0


def compute_answer_correctness(answer: str, expected: str) -> float:
    """
    Compare generated answer against ground truth using embeddings.
    """
    try:
        embeddings = get_mistral_embeddings([answer, expected])
        return cosine_similarity(embeddings[0], embeddings[1])
    except Exception:
        return 0.5  # Default on error


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

@dataclass
class EvaluationResult:
    query: str
    category: str
    answer: str
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    answer_correctness: float


def evaluate_rag_pipeline(test_cases: list[RAGTestCase], vector_store, verbose: bool = True) -> list[EvaluationResult]:
    """Run evaluation on test cases."""
    results = []

    for i, test_case in enumerate(test_cases):
        if verbose:
            print(f"\n[{i+1}/{len(test_cases)}] Evaluating: {test_case.query[:50]}...")

        # Run RAG pipeline
        contexts = vector_store.search(test_case.query, top_k=3)
        time.sleep(0.5)  # Rate limit protection

        answer = generate_answer(test_case.query, contexts)
        time.sleep(0.5)

        # Compute metrics (with delays)
        ctx_rel = compute_context_relevance(test_case.query, contexts)
        time.sleep(0.5)

        faith = compute_faithfulness(answer, contexts)
        time.sleep(0.5)

        ans_rel = compute_answer_relevance(test_case.query, answer)
        time.sleep(0.5)

        ans_corr = compute_answer_correctness(answer, test_case.expected_answer)

        result = EvaluationResult(
            query=test_case.query,
            category=test_case.category,
            answer=answer,
            context_relevance=ctx_rel,
            faithfulness=faith,
            answer_relevance=ans_rel,
            answer_correctness=ans_corr
        )
        results.append(result)

        if verbose:
            print(f"   Context Relevance: {ctx_rel:.2f}")
            print(f"   Faithfulness:      {faith:.2f}")
            print(f"   Answer Relevance:  {ans_rel:.2f}")
            print(f"   Answer Correctness:{ans_corr:.2f}")

    return results


def aggregate_results(results: list[EvaluationResult]) -> dict:
    """Compute aggregate statistics."""
    metrics = {
        "context_relevance": [r.context_relevance for r in results],
        "faithfulness": [r.faithfulness for r in results],
        "answer_relevance": [r.answer_relevance for r in results],
        "answer_correctness": [r.answer_correctness for r in results],
    }

    aggregated = {}
    for name, values in metrics.items():
        aggregated[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    # Category breakdown
    categories = set(r.category for r in results)
    aggregated["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        aggregated["by_category"][cat] = {
            "count": len(cat_results),
            "context_relevance": np.mean([r.context_relevance for r in cat_results]),
            "faithfulness": np.mean([r.faithfulness for r in cat_results]),
            "answer_relevance": np.mean([r.answer_relevance for r in cat_results]),
        }

    return aggregated


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RAGAS-Style Automated RAG Evaluation Pipeline")
    print("=" * 70)

    # Build vector store (ChromaDB - persistent)
    print("\n1. Building vector store (ChromaDB)...")
    vector_store = get_vector_store("ragas_eval")
    vector_store.add_documents(DOCS)
    print(f"   Documents indexed: {vector_store.count()}")

    # Run evaluation
    print(f"\n2. Evaluating {len(TEST_CASES)} test cases...")
    print("   (Adding delays between API calls to avoid rate limits)")
    results = evaluate_rag_pipeline(TEST_CASES, vector_store, verbose=True)

    # Aggregate results
    print("\n" + "=" * 70)
    print("3. AGGREGATE RESULTS")
    print("=" * 70)

    agg = aggregate_results(results)

    print("\n--- Overall Metrics ---")
    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 52)
    for metric in ["context_relevance", "faithfulness", "answer_relevance", "answer_correctness"]:
        stats = agg[metric]
        print(f"{metric:<20} {stats['mean']:>8.2f} {stats['std']:>8.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")

    print("\n--- By Category ---")
    print(f"{'Category':<15} {'Count':>6} {'Ctx Rel':>8} {'Faith':>8} {'Ans Rel':>8}")
    print("-" * 47)
    for cat, stats in agg["by_category"].items():
        print(f"{cat:<15} {stats['count']:>6} {stats['context_relevance']:>8.2f} {stats['faithfulness']:>8.2f} {stats['answer_relevance']:>8.2f}")

    # Decision framework
    print("\n" + "=" * 70)
    print("4. DECISION FRAMEWORK (from lecture)")
    print("=" * 70)

    ctx_rel_mean = agg["context_relevance"]["mean"]
    faith_mean = agg["faithfulness"]["mean"]
    ans_rel_mean = agg["answer_relevance"]["mean"]

    print(f"""
Based on your scores:
- Context Relevance: {ctx_rel_mean:.2f} {"(GOOD)" if ctx_rel_mean >= 0.7 else "(NEEDS IMPROVEMENT)"}
- Faithfulness:      {faith_mean:.2f} {"(GOOD)" if faith_mean >= 0.8 else "(NEEDS IMPROVEMENT)"}
- Answer Relevance:  {ans_rel_mean:.2f} {"(GOOD)" if ans_rel_mean >= 0.7 else "(NEEDS IMPROVEMENT)"}

RECOMMENDATIONS:""")

    if ctx_rel_mean < 0.7:
        print("- LOW Context Relevance: Try HyDE for vocabulary mismatch")
        print("  or improve chunking strategy")

    if faith_mean < 0.8:
        print("- LOW Faithfulness: Implement Self-Corrective RAG")
        print("  or add citation requirements to prompts")

    if ans_rel_mean < 0.7 and faith_mean >= 0.8:
        print("- LOW Answer Relevance but HIGH Faithfulness:")
        print("  This is a GENERATION problem, not retrieval")
        print("  Improve your prompts or fine-tune the model")

    if ctx_rel_mean >= 0.7 and faith_mean >= 0.8 and ans_rel_mean >= 0.7:
        print("- All metrics look good! Basic RAG is sufficient.")
        print("  Only add HyDE/Self-Correction if specific query types underperform.")

    print("""
KEY INSIGHT from lecture:
"You can now make an informed decision: the improvement in retrieval
quality is worth the additional API cost for your use case.
This is evidence-based engineering instead of guessing."
""")


if __name__ == "__main__":
    main()
