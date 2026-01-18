"""
Shared Vector Store Module
==========================

Uses ChromaDB for persistent vector storage.
Embeddings are computed once and cached, avoiding repeated API calls.

Benefits:
- Persisted embeddings (no re-embedding on restart)
- Efficient similarity search
- Reduces API rate limit issues
"""

import os
import time
from dotenv import load_dotenv
from mistralai import Mistral
import chromadb
from chromadb.config import Settings

load_dotenv()

# Initialize Mistral client
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))

# ChromaDB persistence directory
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".chroma_db"
)


def get_mistral_embeddings(texts: list[str], retry_delay: float = 1.0) -> list[list[float]]:
    """
    Get embeddings from Mistral with retry logic for rate limits.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise


class MistralEmbeddingFunction:
    """Custom embedding function for ChromaDB using Mistral."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return get_mistral_embeddings(input)


class ChromaVectorStore:
    """
    ChromaDB-backed vector store with Mistral embeddings.

    Embeddings are persisted to disk, so documents only need to be
    embedded once. Subsequent runs reuse the cached embeddings.
    """

    def __init__(self, collection_name: str = "rag_docs", persist: bool = True):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist: Whether to persist to disk (True) or use in-memory (False)
        """
        if persist:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        else:
            self.chroma_client = chromadb.Client()

        self.embedding_function = MistralEmbeddingFunction()

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.collection_name = collection_name

    def add_documents(self, docs: list[dict], batch_size: int = 5):
        """
        Add documents to the vector store.

        Only adds documents that don't already exist (by ID).
        """
        # Check which docs already exist
        existing_ids = set(self.collection.get()["ids"])
        new_docs = [doc for doc in docs if doc["id"] not in existing_ids]

        if not new_docs:
            print(f"All {len(docs)} documents already indexed.")
            return

        print(f"Indexing {len(new_docs)} new documents...")

        # Process in batches to avoid rate limits
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]

            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]

            # Get embeddings
            embeddings = get_mistral_embeddings(texts)

            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings
            )

            print(f"  Indexed batch {i // batch_size + 1}/{(len(new_docs) + batch_size - 1) // batch_size}")

            # Small delay between batches to avoid rate limits
            if i + batch_size < len(new_docs):
                time.sleep(0.5)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for similar documents.

        Returns list of dicts with id, text, and similarity score.
        """
        # Get query embedding
        query_embedding = get_mistral_embeddings([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances, convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            similarity = 1 - distance

            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "similarity": similarity
            })

        return formatted

    def smart_search(
        self,
        query: str,
        min_threshold: float = 0.80,
        gap_threshold: float = 0.03,
        max_results: int = 10,
        min_results: int = 1
    ) -> list[dict]:
        """
        Smart search using threshold + gap detection.

        Instead of fixed top_k, this method:
        1. Retrieves up to max_results candidates
        2. Filters by min_threshold (minimum similarity score)
        3. Detects "gaps" in similarity scores to find natural cutoff points

        Args:
            query: Search query
            min_threshold: Minimum similarity score to include (default 0.75)
            gap_threshold: If similarity drops by more than this between consecutive
                          docs, stop there (default 0.05 = 5% drop)
            max_results: Maximum documents to return (default 10)
            min_results: Minimum documents to return if above threshold (default 1)

        Returns:
            List of relevant documents with adaptive sizing
        """
        query_embedding = get_mistral_embeddings([query])[0]
        return self.smart_search_with_embedding(
            query_embedding,
            min_threshold=min_threshold,
            gap_threshold=gap_threshold,
            max_results=max_results,
            min_results=min_results
        )

    def smart_search_with_embedding(
        self,
        embedding: list[float],
        min_threshold: float = 0.80,
        gap_threshold: float = 0.03,
        max_results: int = 10,
        min_results: int = 1
    ) -> list[dict]:
        """
        Smart search using a pre-computed embedding (for HyDE).

        Uses threshold + gap detection for adaptive result sizing.
        """
        # Fetch more candidates than we might need
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=max_results,
            include=["documents", "distances"]
        )

        if not results["ids"][0]:
            return []

        # Convert to list of (similarity, doc_info) for processing
        candidates = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            candidates.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "similarity": similarity
            })

        # Apply smart filtering
        filtered = []
        prev_similarity = None

        for i, doc in enumerate(candidates):
            # Check minimum threshold
            if doc["similarity"] < min_threshold:
                break

            # Check for gap (significant drop in similarity)
            if prev_similarity is not None:
                gap = prev_similarity - doc["similarity"]
                if gap > gap_threshold and len(filtered) >= min_results:
                    break

            filtered.append(doc)
            prev_similarity = doc["similarity"]

        # Ensure we return at least min_results if available above threshold
        if len(filtered) < min_results:
            for doc in candidates[:min_results]:
                if doc not in filtered and doc["similarity"] >= min_threshold:
                    filtered.append(doc)

        return filtered

    def search_with_embedding(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Search using a pre-computed embedding (for HyDE).
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "similarity": similarity
            })

        return formatted

    def clear(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {self.collection_name}")

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self.collection.count()


def get_vector_store(collection_name: str = "rag_docs") -> ChromaVectorStore:
    """
    Get a vector store instance.

    This is the main entry point for other scripts.
    """
    return ChromaVectorStore(collection_name=collection_name)


# For backwards compatibility - simple wrapper
class SimpleVectorStore(ChromaVectorStore):
    """Alias for ChromaVectorStore for backwards compatibility."""

    def __init__(self):
        super().__init__(collection_name="rag_docs", persist=True)


if __name__ == "__main__":
    # Test the vector store
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from toy_corpus import DOCS

    print("Testing ChromaDB Vector Store")
    print("=" * 50)

    store = get_vector_store("test_collection")

    print(f"\nCurrent document count: {store.count()}")

    print("\nAdding documents...")
    store.add_documents(DOCS)

    print(f"\nDocument count after add: {store.count()}")

    query = "How do I configure SSL for production?"

    print(f"\n{'='*50}")
    print("COMPARISON: Fixed top_k vs Smart Search")
    print(f"{'='*50}")

    print(f"\n--- Fixed top_k=3 (old method) ---")
    results = store.search(query, top_k=3)
    for r in results:
        print(f"  [{r['id']}] (sim={r['similarity']:.3f}) {r['text'][:50]}...")

    print(f"\n--- Smart Search (threshold + gap detection) ---")
    smart_results = store.smart_search(query)
    for r in smart_results:
        print(f"  [{r['id']}] (sim={r['similarity']:.3f}) {r['text'][:50]}...")

    print(f"\nResults: Fixed={len(results)}, Smart={len(smart_results)}")
    print("\nDone!")
