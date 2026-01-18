# vector_store.py - Shared ChromaDB Vector Store

## What This File Does

This is the shared vector storage module used by all lecture files. It provides a persistent, efficient way to store and search document embeddings using ChromaDB.

## Why ChromaDB?

The original lecture files used in-memory storage:
```python
# Old approach - embeddings lost on restart
class SimpleVectorStore:
    def __init__(self):
        self.documents = []      # Lost when program ends
        self.embeddings = []     # Re-computed every run
```

With ChromaDB:
```python
# New approach - embeddings persist to disk
class ChromaVectorStore:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=".chroma_db")
        # Embeddings saved forever, never re-computed
```

**Benefits:**
- **Persistence** - Embeddings saved to disk, survive restarts
- **No re-embedding** - Documents indexed once, reused forever
- **Rate limit friendly** - Fewer API calls needed
- **Production-ready** - Same tech used in real applications

## How to Use It

### Basic Usage

```python
from lecture.vector_store import get_vector_store

# Create or load a vector store
store = get_vector_store(collection_name="my_rag_app")

# Add documents (only embeds new ones)
docs = [
    {"id": "doc1", "text": "SSL certificates use PEM format..."},
    {"id": "doc2", "text": "Configure HTTPS on port 443..."},
]
store.add_documents(docs)

# Search
results = store.search("How do I set up SSL?", top_k=3)
for r in results:
    print(f"{r['id']}: {r['similarity']:.2f}")
```

### Search with Pre-computed Embedding (for HyDE)

```python
from lecture.vector_store import get_vector_store, get_mistral_embeddings

store = get_vector_store("hyde_rag")

# Generate hypothetical and get its embedding
hypothetical = "SSL certificates should be stored in /etc/ssl/..."
embedding = get_mistral_embeddings([hypothetical])[0]

# Search using that embedding
results = store.search_with_embedding(embedding, top_k=3)
```

## Key Functions

### get_vector_store()

```python
def get_vector_store(collection_name: str = "rag_docs") -> ChromaVectorStore:
```

Creates or loads a ChromaDB collection. Each collection is isolated - different RAG applications can use different collections.

**Example:**
```python
baseline_store = get_vector_store("baseline_rag")
hyde_store = get_vector_store("hyde_rag")
# These are separate, don't interfere with each other
```

### get_mistral_embeddings()

```python
def get_mistral_embeddings(texts: list[str]) -> list[list[float]]:
```

Gets embeddings from Mistral with automatic retry for rate limits.

**Features:**
- Batch processing (multiple texts at once)
- Exponential backoff on rate limits
- Returns list of embedding vectors

## The ChromaVectorStore Class

### add_documents()

```python
def add_documents(self, docs: list[dict], batch_size: int = 5):
```

Adds documents to the store. **Only indexes new documents** - if a document ID already exists, it's skipped.

```python
# First run: all 7 docs are embedded
store.add_documents(DOCS)
# Output: "Indexing 7 new documents..."

# Second run: nothing to do
store.add_documents(DOCS)
# Output: "All 7 documents already indexed."
```

**Batching:**
Processes 5 documents at a time with delays to avoid rate limits.

### search()

```python
def search(self, query: str, top_k: int = 3) -> list[dict]:
```

Finds the most similar documents to a query.

**Returns:**
```python
[
    {"id": "doc1", "text": "...", "similarity": 0.85},
    {"id": "doc2", "text": "...", "similarity": 0.78},
    {"id": "doc3", "text": "...", "similarity": 0.72},
]
```

Similarity is 0-1, where 1 = identical meaning.

### search_with_embedding()

```python
def search_with_embedding(self, embedding: list[float], top_k: int = 3) -> list[dict]:
```

Search using a pre-computed embedding. Used for HyDE where you embed the hypothetical document, not the query.

### clear()

```python
def clear(self):
```

Deletes all documents from the collection. Useful for testing or starting fresh.

### count()

```python
def count(self) -> int:
```

Returns the number of documents in the collection.

## Under the Hood

### Storage Location

```python
CHROMA_PERSIST_DIR = ".chroma_db"  # In project root
```

All data is stored in a `.chroma_db` folder. Add this to `.gitignore`.

### Cosine Similarity

```python
self.collection = self.chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
```

ChromaDB uses cosine similarity by default, which measures the angle between vectors (same as what we taught in the embeddings lecture).

### Custom Embedding Function

```python
class MistralEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return get_mistral_embeddings(input)
```

ChromaDB can auto-embed documents, but we use a custom function to ensure Mistral embeddings.

## Common Patterns

### Different Collections per Lecture

```python
# Each lecture uses its own collection
baseline_store = get_vector_store("baseline_rag")
hyde_store = get_vector_store("hyde_rag")
eval_store = get_vector_store("evaluation_rag")
```

### Check if Documents Exist

```python
store = get_vector_store("my_app")
if store.count() == 0:
    print("First run - indexing documents...")
    store.add_documents(docs)
else:
    print(f"Using {store.count()} cached documents")
```

### Start Fresh

```python
store = get_vector_store("test_collection")
store.clear()  # Delete everything
store.add_documents(new_docs)  # Re-index
```

## Why This Matters

**Without persistence:**
- Every run re-embeds all documents
- Hits API rate limits constantly
- Slow startup times
- Wastes money on redundant embeddings

**With ChromaDB:**
- Documents embedded once, forever
- Fast startup (just load from disk)
- Rate limits only for queries
- Production-ready architecture
