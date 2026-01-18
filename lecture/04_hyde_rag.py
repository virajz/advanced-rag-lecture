"""
Step 4: HyDE RAG (Hypothetical Document Embeddings)
====================================================

HyDE solves the vocabulary mismatch problem in RAG.

THE PROBLEM:
- User asks: "Why is my app crashing?"
- Docs say: "Application failures occur due to missing config..."
- Different words = poor embedding similarity = bad retrieval

THE SOLUTION (HyDE):
1. Ask LLM to generate a HYPOTHETICAL answer (before searching)
2. Embed that hypothetical answer (not the question)
3. Search using the hypothetical embedding
4. The hypothetical uses document-style language = better matches!

Run: uv run python lecture/04_hyde_rag.py
"""

import os
from dotenv import load_dotenv
from mistralai import Mistral

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS

# Import ChromaDB-backed vector store
from lecture.vector_store import get_vector_store, get_mistral_embeddings

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# =============================================================================
# HyDE COMPONENTS
# =============================================================================

def generate_hypothetical_document(query: str) -> str:
    """
    Generate a hypothetical document that WOULD answer the query.

    This is the key insight of HyDE:
    - Questions are short and use user vocabulary
    - Documents are detailed and use technical vocabulary
    - By generating a hypothetical answer, we bridge this gap
    """
    prompt = f"""You are a technical documentation writer.
Given a user's question, write a short technical document (2-3 sentences)
that would answer this question. Write in a factual, documentation style.
Do NOT say "this document explains" - just write the actual content.

Question: {query}

Technical document:"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def hyde_retrieve(query: str, vector_store, top_k: int = 3) -> dict:
    """
    HyDE retrieval: generate hypothetical doc first, then search with it.

    Returns both the hypothetical doc and search results for inspection.
    """
    # Step 1: Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query)

    # Step 2: Embed the hypothetical document (NOT the query)
    hyde_embedding = get_mistral_embeddings([hypothetical_doc])[0]

    # Step 3: Search using the hypothetical embedding
    results = vector_store.search_with_embedding(hyde_embedding, top_k)

    return {
        "hypothetical_doc": hypothetical_doc,
        "results": results
    }


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
# COMPARISON: BASELINE vs HyDE
# =============================================================================

def compare_baseline_vs_hyde(query: str, vector_store):
    """
    Compare baseline retrieval vs HyDE retrieval for the same query.
    """
    print(f"\nQUERY: {query}")
    print("=" * 60)

    # Baseline retrieval
    print("\n--- BASELINE RETRIEVAL ---")
    print("(Embed the question directly)")
    baseline_results = vector_store.search(query, top_k=3)
    for ctx in baseline_results:
        print(f"  [{ctx['id']}] sim={ctx['similarity']:.4f}")
        print(f"    {ctx['text'][:60]}...")

    # HyDE retrieval
    print("\n--- HyDE RETRIEVAL ---")
    hyde_result = hyde_retrieve(query, vector_store, top_k=3)

    print(f"Hypothetical document generated:")
    print(f"  \"{hyde_result['hypothetical_doc'][:100]}...\"")
    print("\nSearch results using hypothetical embedding:")
    for ctx in hyde_result["results"]:
        print(f"  [{ctx['id']}] sim={ctx['similarity']:.4f}")
        print(f"    {ctx['text'][:60]}...")

    return baseline_results, hyde_result["results"]


# =============================================================================
# FULL HyDE RAG PIPELINE
# =============================================================================

def hyde_rag(query: str, vector_store, top_k: int = 3) -> dict:
    """
    Complete HyDE RAG pipeline:
    1. Generate hypothetical document
    2. Retrieve using hypothetical embedding
    3. Generate answer from retrieved context
    """
    # HyDE retrieval
    hyde_result = hyde_retrieve(query, vector_store, top_k)

    # Generate answer
    answer = generate_answer(query, hyde_result["results"])

    return {
        "query": query,
        "hypothetical_doc": hyde_result["hypothetical_doc"],
        "contexts": hyde_result["results"],
        "answer": answer
    }


def main():
    print("=" * 60)
    print("HyDE RAG (Hypothetical Document Embeddings)")
    print("=" * 60)

    # Build vector store (using ChromaDB)
    print("\n1. Building vector store with ChromaDB...")
    vector_store = get_vector_store(collection_name="hyde_rag")
    vector_store.add_documents(DOCS)
    print(f"   Indexed {len(DOCS)} documents")

    # Compare on the tricky query from baseline
    print("\n2. Comparing Baseline vs HyDE on vocabulary mismatch query...")
    compare_baseline_vs_hyde(
        "Why is my app crashing on startup?",
        vector_store
    )

    # More comparisons
    print("\n" + "=" * 60)
    print("3. More comparisons...")

    queries = [
        "How do I make my website secure?",  # Doesn't mention SSL/TLS
        "What files do I need for HTTPS?",   # Should find cert paths
    ]

    for query in queries:
        compare_baseline_vs_hyde(query, vector_store)

    # Full HyDE RAG demo
    print("\n" + "=" * 60)
    print("4. Full HyDE RAG Pipeline Demo")
    print("=" * 60)

    query = "How do I set up encryption for my production server?"
    result = hyde_rag(query, vector_store, top_k=3)

    print(f"\nQUERY: {query}")
    print(f"\nHYPOTHETICAL DOC:\n  {result['hypothetical_doc']}")
    print("\nRETRIEVED CONTEXTS:")
    for ctx in result["contexts"]:
        print(f"  [{ctx['id']}] {ctx['text'][:60]}...")
    print(f"\nFINAL ANSWER:\n{result['answer']}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. HyDE generates a hypothetical answer BEFORE searching
2. This bridges the vocabulary gap between questions and documents
3. "encryption" in query -> hypothetical mentions "SSL", "certificates"
4. Better retrieval = better final answers

TRADE-OFF:
- Extra LLM call adds latency and cost
- Hypothetical might be wrong (but retrieval corrects it)

WHEN TO USE HyDE:
- Short, vague queries
- Vocabulary mismatch between users and documents
- FAQ/support systems where users don't know technical terms

NEXT: 05_self_corrective_rag.py - Reflect on retrieval quality
""")


if __name__ == "__main__":
    main()
