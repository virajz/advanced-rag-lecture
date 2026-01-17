"""
Step 2: Understanding Embeddings with Mistral AI
=================================================

Embeddings are vector representations of text that capture semantic meaning.
Similar texts have similar vectors (close in vector space).

This script teaches:
1. How to create embeddings with Mistral
2. How cosine similarity works
3. How to build a simple semantic search

Run: uv run python lecture/02_embeddings.py
"""

import os
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# Initialize Mistral client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def get_embedding(text: str) -> list[float]:
    """
    Convert text to a vector embedding using Mistral.

    The embedding captures the semantic meaning of the text.
    Similar meanings = similar vectors.
    """
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for multiple texts in one API call.
    More efficient than calling one at a time.
    """
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns a value between -1 and 1:
    - 1.0 = identical direction (very similar)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (opposite meaning)

    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)


def demo_embedding_basics():
    """Demonstrate basic embedding concepts."""
    print("=" * 60)
    print("PART 1: Embedding Basics")
    print("=" * 60)

    text = "How do I configure SSL certificates?"
    embedding = get_embedding(text)

    print(f"\nText: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")


def demo_similarity():
    """Demonstrate how similarity captures semantic meaning."""
    print("\n" + "=" * 60)
    print("PART 2: Semantic Similarity")
    print("=" * 60)

    # Define test sentences
    sentences = [
        "How do I configure SSL certificates?",      # Original query
        "Setting up HTTPS for my website",           # Similar meaning, different words
        "SSL certificate configuration guide",       # Similar meaning
        "What's the weather like today?",            # Completely different topic
        "I love eating pizza on weekends",           # Unrelated
    ]

    print("\nComparing sentences to: 'How do I configure SSL certificates?'\n")

    # Get all embeddings in one batch call
    embeddings = get_embeddings_batch(sentences)
    query_embedding = embeddings[0]

    # Calculate similarities
    results = []
    for i, (sentence, embedding) in enumerate(zip(sentences[1:], embeddings[1:]), 1):
        similarity = cosine_similarity(query_embedding, embedding)
        results.append((similarity, sentence))

    # Sort by similarity (highest first)
    results.sort(reverse=True)

    print("Similarity scores (sorted):")
    print("-" * 50)
    for similarity, sentence in results:
        # Visual indicator of similarity
        bar = "█" * int(similarity * 20)
        print(f"{similarity:.4f} {bar}")
        print(f"         '{sentence}'\n")


def demo_semantic_search():
    """Build a simple semantic search over documents."""
    print("\n" + "=" * 60)
    print("PART 3: Semantic Search (Mini RAG)")
    print("=" * 60)

    # Our mini knowledge base
    documents = [
        {"id": "doc1", "text": "SSL certificates use PEM or DER encoding formats."},
        {"id": "doc2", "text": "To configure HTTPS, set the certificate path in config."},
        {"id": "doc3", "text": "Python is a popular programming language."},
        {"id": "doc4", "text": "Rotate certificates every 90 days for security."},
        {"id": "doc5", "text": "The weather forecast shows rain tomorrow."},
    ]

    query = "How do I set up SSL?"

    print(f"\nQuery: '{query}'")
    print(f"Searching {len(documents)} documents...\n")

    # Get embeddings for all documents and query
    doc_texts = [doc["text"] for doc in documents]
    all_texts = [query] + doc_texts
    all_embeddings = get_embeddings_batch(all_texts)

    query_embedding = all_embeddings[0]
    doc_embeddings = all_embeddings[1:]

    # Calculate similarities and rank
    results = []
    for doc, embedding in zip(documents, doc_embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "similarity": similarity
        })

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    print("Search Results (ranked by similarity):")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        sim = result["similarity"]
        relevance = "HIGH" if sim > 0.7 else "MEDIUM" if sim > 0.5 else "LOW"
        print(f"{i}. [{relevance}] (sim={sim:.4f})")
        print(f"   {result['text']}\n")

    print("\nNotice how:")
    print("- SSL/HTTPS docs rank highest (semantic match)")
    print("- Certificate rotation is relevant (same domain)")
    print("- Weather/Python docs rank lowest (different topics)")


def main():
    print("Embeddings Tutorial with Mistral AI")
    print("=" * 60)

    demo_embedding_basics()
    demo_similarity()
    demo_semantic_search()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. Embeddings convert text to vectors (1024 dimensions for Mistral)
2. Similar meanings = similar vectors (high cosine similarity)
3. This enables SEMANTIC search, not just keyword matching
4. "SSL" and "HTTPS" match well even though they're different words!

Next step: Use this for RAG in 03_baseline_rag.py
""")


if __name__ == "__main__":
    main()
