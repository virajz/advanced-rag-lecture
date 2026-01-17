"""
Step 3: Baseline RAG Implementation
====================================

This implements a complete RAG (Retrieval-Augmented Generation) system:
1. Embed all documents in our corpus
2. Embed the user query
3. Find most similar documents (retrieval)
4. Generate an answer using the retrieved context

This is the "vanilla" RAG that we'll improve upon in later steps.

Run: uv run python lecture/03_baseline_rag.py
"""

import os
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

# Import our toy corpus
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS

load_dotenv()

# Initialize Mistral client
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
# RAG COMPONENTS
# =============================================================================

class SimpleVectorStore:
    """
    A simple in-memory vector store.

    In production, you'd use a real vector DB (Pinecone, Chroma, etc.)
    but this demonstrates the core concept.
    """

    def __init__(self):
        self.documents = []      # Original documents
        self.embeddings = []     # Their embeddings

    def add_documents(self, docs: list[dict]):
        """Add documents and compute their embeddings."""
        texts = [doc["text"] for doc in docs]
        embeddings = get_embeddings_batch(texts)

        for doc, embedding in zip(docs, embeddings):
            self.documents.append(doc)
            self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most similar documents to the query.

        Returns documents with their similarity scores.
        """
        # Embed the query
        query_embedding = get_embeddings_batch([query])[0]

        # Calculate similarity with all documents
        results = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "similarity": similarity
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


def generate_answer(query: str, contexts: list[dict]) -> str:
    """
    Generate an answer using the LLM with retrieved context.

    This is the "Generation" part of RAG.
    """
    # Format the context
    context_text = "\n\n".join([
        f"[{ctx['id']}]: {ctx['text']}"
        for ctx in contexts
    ])

    # Create the prompt
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    # Call the LLM
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# =============================================================================
# MAIN RAG PIPELINE
# =============================================================================

def baseline_rag(query: str, vector_store: SimpleVectorStore, top_k: int = 3) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Generate answer from context

    Returns the answer and retrieved contexts for inspection.
    """
    # Step 1: Retrieve
    contexts = vector_store.search(query, top_k=top_k)

    # Step 2: Generate
    answer = generate_answer(query, contexts)

    return {
        "query": query,
        "contexts": contexts,
        "answer": answer
    }


def main():
    print("=" * 60)
    print("Baseline RAG with Mistral AI")
    print("=" * 60)

    # Step 1: Build the vector store
    print("\n1. Building vector store...")
    print(f"   Indexing {len(DOCS)} documents from toy_corpus.py")

    vector_store = SimpleVectorStore()
    vector_store.add_documents(DOCS)
    print("   Done!")

    # Step 2: Run some queries
    queries = [
        "How do I configure SSL certificates for production?",
        "What port should I use for containerized deployments?",
        "What are the compliance requirements for TLS?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"QUERY {i}: {query}")
        print("=" * 60)

        result = baseline_rag(query, vector_store, top_k=3)

        # Show retrieved contexts
        print("\nRETRIEVED DOCUMENTS:")
        for ctx in result["contexts"]:
            print(f"  [{ctx['id']}] (sim={ctx['similarity']:.4f})")
            print(f"    {ctx['text'][:80]}...")

        # Show answer
        print(f"\nANSWER:\n{result['answer']}")

    # Demonstrate the limitation
    print("\n" + "=" * 60)
    print("DEMONSTRATING RAG LIMITATION")
    print("=" * 60)

    # This query has vocabulary mismatch
    tricky_query = "Why is my app crashing on startup?"
    print(f"\nTRICKY QUERY: {tricky_query}")
    print("(This query doesn't use SSL/TLS terminology)")

    result = baseline_rag(tricky_query, vector_store, top_k=3)

    print("\nRETRIEVED DOCUMENTS:")
    for ctx in result["contexts"]:
        print(f"  [{ctx['id']}] (sim={ctx['similarity']:.4f})")

    print(f"\nANSWER:\n{result['answer']}")

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("""
1. Baseline RAG works well when query terms match document terms
2. It struggles with vocabulary mismatch (different words, same meaning)
3. It retrieves based on similarity, not relevance to answering

NEXT STEPS:
- 04_hyde_rag.py: Generate hypothetical answer first (fixes vocab mismatch)
- 05_self_corrective_rag.py: Reflect and retry if retrieval is poor
""")


if __name__ == "__main__":
    main()
