# 03_baseline_rag.py - Baseline RAG Implementation

## What This File Does

This file implements a complete RAG (Retrieval-Augmented Generation) system from scratch. It shows the basic pattern that all RAG systems follow.

## The RAG Pipeline (3 Steps)

### Step 1: Index Documents (One-Time Setup)

```python
vector_store = get_vector_store(collection_name="baseline_rag")
vector_store.add_documents(DOCS)
```

**What's happening:**
- Load your knowledge base (documents, FAQs, documentation)
- Convert each document into an embedding (vector of numbers)
- Store them in ChromaDB for fast searching

**Think of it like:** Building a library's card catalog. You're organizing all your books so you can find them quickly later.

### Step 2: Retrieve Relevant Documents

```python
contexts = vector_store.search(query, top_k=3)
```

**What's happening:**
- User asks a question
- Convert the question to an embedding
- Find the 3 most similar documents in the database
- Return those documents as "context"

**Think of it like:** A librarian who reads your question, understands what you need, and brings you the 3 most relevant books.

### Step 3: Generate Answer

```python
def generate_answer(query: str, contexts: list[dict]) -> str:
    prompt = f"""Answer based ONLY on the context.
    CONTEXT: {context_text}
    QUESTION: {query}
    ANSWER:"""

    response = client.chat.complete(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content
```

**What's happening:**
- Combine the user's question with the retrieved documents
- Send this to the LLM with instructions to only use the provided context
- The LLM generates an answer based on the documents

**Think of it like:** Giving a student some reference materials and asking them to answer a question using only those materials. They can't make things up - they must cite their sources.

## The Complete Pipeline

```python
def baseline_rag(query: str, vector_store, top_k: int = 3) -> dict:
    # Step 1: Retrieve
    contexts = vector_store.search(query, top_k=top_k)

    # Step 2: Generate
    answer = generate_answer(query, contexts)

    return {"query": query, "contexts": contexts, "answer": answer}
```

That's it! This is the foundation of all RAG systems.

## Key Concepts Explained

### Why Use ChromaDB?

```python
from lecture.vector_store import get_vector_store
vector_store = get_vector_store(collection_name="baseline_rag")
```

ChromaDB is a vector database that:
- **Persists data** - Embeddings are saved to disk, not re-computed every run
- **Fast search** - Optimized for finding similar vectors quickly
- **Handles scale** - Works with thousands or millions of documents

### Why "top_k=3"?

We retrieve the top 3 most similar documents. This is a balance:
- Too few = might miss relevant information
- Too many = adds noise and costs more tokens

3-5 is a common starting point; adjust based on your use case.

### The Prompt Pattern

```python
prompt = """Answer based ONLY on the context.
If the context doesn't contain enough information, say so.

CONTEXT: {documents}
QUESTION: {query}
ANSWER:"""
```

This structure:
- Tells the LLM to stay grounded (no hallucination)
- Provides the retrieved information
- Asks for an answer

## Limitations of Baseline RAG

The file demonstrates a key problem:

**Query:** "Why is my app crashing on startup?"

This query fails because:
- User says "crashing" and "app"
- Documents say "SSL," "TLS," "certificates"
- Different words = low embedding similarity = wrong documents retrieved

**This is called vocabulary mismatch.** The user and the documents use different words for the same concepts.

The next lectures (HyDE, Self-Corrective) solve this problem.

## When to Use Baseline RAG

Baseline RAG works great when:
- Users know the right terminology
- Queries are specific and detailed
- Documents use consistent vocabulary

It struggles when:
- Users are vague or use different words
- The question requires combining info from multiple documents
- There's significant vocabulary mismatch
