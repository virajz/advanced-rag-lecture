"""
Step 5: Self-Corrective RAG
============================

Self-Corrective RAG adds a "reflection" step that checks if retrieved
documents are good enough to answer the question. If not, it reformulates
the query and tries again.

THE PROBLEM:
- Sometimes the first retrieval doesn't get the right documents
- Baseline RAG just uses whatever it gets (even if irrelevant)
- This leads to poor or hallucinated answers

THE SOLUTION:
1. Retrieve documents
2. REFLECT: "Do these docs answer the question?" (LLM judges)
3. If NO: Reformulate query with better keywords, go back to step 1
4. If YES: Generate answer

Run: uv run python lecture/05_self_corrective_rag.py
"""

import os
from dotenv import load_dotenv
from mistralai import Mistral

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS

# Import ChromaDB-backed vector store
from lecture.vector_store import get_vector_store

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# =============================================================================
# SELF-CORRECTIVE COMPONENTS
# =============================================================================

def reflect(query: str, contexts: list[dict]) -> dict:
    """
    Reflect on whether the retrieved contexts are sufficient to answer the query.

    This is the "judge" that decides if we need to retry retrieval.

    Returns:
        {"decision": "YES" or "NO", "reason": "explanation"}
    """
    context_text = "\n".join([
        f"- [{ctx['id']}]: {ctx['text']}"
        for ctx in contexts
    ])

    prompt = f"""You are evaluating whether retrieved documents can answer a question.

QUESTION: {query}

RETRIEVED DOCUMENTS:
{context_text}

Can these documents provide a complete, accurate answer to the question?
Consider:
- Do they contain specific, actionable information?
- Do they directly address what the user is asking?
- Is there enough detail to give a helpful answer?

Respond in this exact format:
DECISION: YES or NO
REASON: <one sentence explanation>"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content

    # Parse the response
    decision = "YES" if "DECISION: YES" in response_text.upper() else "NO"

    # Extract reason
    reason = ""
    if "REASON:" in response_text.upper():
        reason = response_text.split("REASON:")[-1].strip()

    return {"decision": decision, "reason": reason}


def reformulate_query(original_query: str, reflection_reason: str) -> str:
    """
    Create a better query based on what was missing in the first retrieval.

    Uses the reflection reason to guide the reformulation.
    """
    prompt = f"""The following search query did not retrieve good enough documents.

ORIGINAL QUERY: {original_query}
PROBLEM: {reflection_reason}

Create a better, more specific search query that might find more relevant documents.
Add specific technical terms, file names, or configuration keywords that might help.

Return ONLY the new query, nothing else."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def generate_answer(query: str, contexts: list[dict]) -> str:
    """Generate an answer using the LLM with retrieved context."""
    context_text = "\n\n".join([
        f"[{ctx['id']}]: {ctx['text']}"
        for ctx in contexts
    ])

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

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
# SELF-CORRECTIVE RAG PIPELINE
# =============================================================================

def self_corrective_rag(
    query: str,
    vector_store,
    max_iterations: int = 3,
    top_k: int = 3,
    verbose: bool = True
) -> dict:
    """
    Self-Corrective RAG pipeline with reflection and query reformulation.

    Flow:
    1. Retrieve documents
    2. Reflect: are they good enough?
    3. If NO and iterations remain: reformulate query, go to step 1
    4. If YES or max iterations reached: generate answer

    Returns detailed trace of the process for inspection.
    """
    trace = {
        "original_query": query,
        "iterations": [],
        "final_answer": None
    }

    current_query = query

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Query: {current_query}")

        # Step 1: Retrieve
        contexts = vector_store.search(current_query, top_k=top_k)

        if verbose:
            print(f"Retrieved: {[ctx['id'] for ctx in contexts]}")

        # Step 2: Reflect
        reflection = reflect(query, contexts)  # Always reflect on ORIGINAL query

        iteration_data = {
            "query": current_query,
            "contexts": contexts,
            "reflection": reflection
        }
        trace["iterations"].append(iteration_data)

        if verbose:
            print(f"Reflection: {reflection['decision']} - {reflection['reason']}")

        # Step 3: Decide next action
        if reflection["decision"] == "YES":
            # Good enough! Generate answer
            if verbose:
                print("Sufficient context found. Generating answer...")

            answer = generate_answer(query, contexts)
            trace["final_answer"] = answer
            trace["final_contexts"] = contexts
            return trace

        # Not good enough - reformulate if iterations remain
        if iteration < max_iterations - 1:
            current_query = reformulate_query(current_query, reflection["reason"])
            if verbose:
                print(f"Reformulated query: {current_query}")
        else:
            if verbose:
                print("Max iterations reached. Generating answer with best available context...")

    # Max iterations reached - generate with what we have
    best_contexts = trace["iterations"][-1]["contexts"]
    answer = generate_answer(query, best_contexts)
    trace["final_answer"] = answer
    trace["final_contexts"] = best_contexts

    return trace


# =============================================================================
# DEMONSTRATION
# =============================================================================

def main():
    print("=" * 60)
    print("Self-Corrective RAG with Reflection")
    print("=" * 60)

    # Build vector store (using ChromaDB)
    print("\n1. Building vector store with ChromaDB...")
    vector_store = get_vector_store(collection_name="self_corrective_rag")
    vector_store.add_documents(DOCS)
    print(f"   Indexed {len(DOCS)} documents")

    # Demo 1: Query that should trigger correction (asks for something specific not directly in docs)
    print("\n" + "=" * 60)
    print("DEMO 1: Multi-hop query (needs info from multiple docs)")
    print("=" * 60)

    # This query needs BOTH compliance requirements AND production steps
    query1 = "What exact port and TLS version should I use for a compliant production container deployment?"

    result1 = self_corrective_rag(query1, vector_store, max_iterations=3, verbose=True)

    print(f"\nFINAL ANSWER:\n{result1['final_answer']}")
    print(f"Iterations needed: {len(result1['iterations'])}")

    # Demo 2: Query that works on first try (direct match)
    print("\n" + "=" * 60)
    print("DEMO 2: Direct query (works immediately)")
    print("=" * 60)

    query2 = "What is the certificate rotation policy?"

    result2 = self_corrective_rag(query2, vector_store, max_iterations=3, verbose=True)

    print(f"\nFINAL ANSWER:\n{result2['final_answer']}")
    print(f"Iterations needed: {len(result2['iterations'])}")

    # Demo 3: Very vague query
    print("\n" + "=" * 60)
    print("DEMO 3: Vague query (may need refinement)")
    print("=" * 60)

    # Intentionally vague - doesn't specify what aspect of "errors"
    query3 = "How do I fix errors?"

    result3 = self_corrective_rag(query3, vector_store, max_iterations=3, verbose=True)

    print(f"\nFINAL ANSWER:\n{result3['final_answer']}")
    print(f"Iterations needed: {len(result3['iterations'])}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. REFLECT: LLM judges if retrieved docs are sufficient
2. REFORMULATE: If not, create a better query with more specific terms
3. ITERATE: Try up to max_iterations times before giving up

WHEN SELF-CORRECTION HELPS:
- Vague or ambiguous queries
- When first retrieval misses key documents
- Complex questions needing specific information

TRADE-OFFS:
- More LLM calls = higher latency and cost
- Reflection might be wrong (false positives/negatives)
- Typically cap at 2-3 iterations in production

COST ANALYSIS:
- Baseline RAG: 1 embedding + 1 generation = 2 API calls
- Self-Corrective (2 iterations): 2 embeddings + 2 reflections + 1 generation = 5 API calls

NEXT: 06_evaluation.py - Measure RAG quality with metrics
""")


if __name__ == "__main__":
    main()
